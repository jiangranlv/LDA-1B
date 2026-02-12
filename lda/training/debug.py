# Copyright 2025 LDA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

"""
LDA Single-GPU Debug Trainer (Pure PyTorch)
- Zero dependency on Accelerate/DeepSpeed/Distributed
- Pure native PyTorch implementation
- Enhanced debugging features (NaN checks, gradient inspection, etc.)
- Minimalist design for rapid iteration
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import time
import re
import sys
import random

# Third-Party Libraries
import torch
import torch.nn as nn
import wandb
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from LDA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from LDA.model.framework import build_framework
from LDA.training.trainer_utils.trainer_tools import TrainerUtils
from LDA.training.trainer_utils.trainer_tools import build_param_lr_groups
from LDA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

# Setup logging
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_directories(cfg) -> Path:
    """Create output directory and save config (single process only)"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "checkpoints", exist_ok=True)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")
    with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
        yaml_cfg = yaml.safe_load(f_yaml)
        json.dump(yaml_cfg, f_json, indent=2)
    
    logger.info(f"📁 Output directory created: {output_dir}")
    return output_dir


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🌱 Random seed set to {seed}")


def prepare_data(cfg) -> Tuple:
    """Prepare training data (single process, no multiprocessing)"""
    logger.info(f"🔍 Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    
    # Force debug-friendly settings
    if cfg.is_debug:
        cfg.datasets.vla_data.per_device_batch_size = min(2, cfg.datasets.vla_data.per_device_batch_size)
        cfg.datasets.vla_data.num_workers = 0  # Disable multiprocessing for easy debugging
        cfg.trainer.max_train_steps = min(50, cfg.trainer.max_train_steps)
        logger.warning(f"⚠️ DEBUG MODE: batch_size={cfg.datasets.vla_data.per_device_batch_size}, max_steps={cfg.trainer.max_train_steps}")

    from LDA.dataloader import build_dataloader, build_multi_task_dataloader
    
    if cfg.framework.name == "QwenGR00T":
        vla_train_dataloader = build_dataloader(
            cfg=cfg, 
            dataset_py=cfg.datasets.vla_data.dataset_py,
            num_workers=0  # Critical for debugging
        )
        return vla_train_dataloader, None, None, None
    else:
        policy_train_dataloader, fd_train_dataloader, vg_train_dataloader, id_train_dataloader = build_multi_task_dataloader(
            cfg=cfg,
            dataset_py=cfg.datasets.vla_data.dataset_py,
        )
        return policy_train_dataloader, fd_train_dataloader, vg_train_dataloader, id_train_dataloader
       

def setup_optimizer_and_scheduler(model: nn.Module, cfg) -> Tuple:
    """Set optimizer and scheduler (pure PyTorch)"""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # Print optimizer groups
    for i, group in enumerate(optimizer.param_groups):
        logger.info(f"⚙️ LR Group {group.get('name', f'group_{i}')}: lr={group['lr']:.2e}, num_params={sum(p.numel() for p in group['params'])}")

    # Scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model: nn.Module, policy_train_dataloader, fd_train_dataloader, 
                 vg_train_dataloader, id_train_dataloader, optimizer, lr_scheduler):
        self.config = cfg
        self.model = model
        self.policy_train_dataloader = policy_train_dataloader
        self.fd_train_dataloader = fd_train_dataloader
        self.vg_train_dataloader = vg_train_dataloader
        self.id_train_dataloader = id_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"💻 Using device: {self.device}")
        if self.device.type == "cuda":
            logger.info(f"   GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

        # Training state
        self.completed_steps = 0
        self.gradient_accumulation_steps = cfg.trainer.gradient_accumulation_steps
        self.total_batch_size = cfg.datasets.vla_data.per_device_batch_size * self.gradient_accumulation_steps
        
        # Move model to device
        self.model = self.model.to(self.device)
        logger.info(f"📦 Model moved to {self.device}")

    def prepare_training(self):
        """Prepare for training (single GPU)"""
        # Set seed
        seed = 42 if self.config.is_debug else (self.config.seed if hasattr(self.config, "seed") else 3047)
        set_seed(seed)

        # Handle checkpoint loading
        self._init_checkpointing()

        # Freeze modules
        freeze_modules = getattr(self.config.trainer, "freeze_modules", None)
        if freeze_modules:
            self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
            logger.info(f"🔒 Frozen modules: {freeze_modules}")

    def _init_checkpointing(self):
        """Initialize checkpoint handling"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if is_resume:
            ckpt_files = sorted(Path(self.checkpoint_dir).glob("steps_*_pytorch_model.pt"))
            if ckpt_files:
                checkpoint_path = str(ckpt_files[-1])
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", checkpoint_path).group(1))
                self._load_model_weights(checkpoint_path)
                logger.info(f"↩️ Resuming from checkpoint: {checkpoint_path} (step {self.completed_steps})")
                return
            else:
                logger.warning(f"⚠️ No checkpoint found in {self.checkpoint_dir}. Starting from scratch.")

        if pretrained_checkpoint:
            self._load_model_weights(pretrained_checkpoint)
            try:
                self.completed_steps = int(re.search(r"steps_(\d+)_pytorch_model\.pt", pretrained_checkpoint).group(1))
            except (AttributeError, ValueError):
                logger.warning(f"⚠️ Could not parse steps from checkpoint: {pretrained_checkpoint}")
                self.completed_steps = 0
            logger.info(f"📦 Loaded pretrained checkpoint: {pretrained_checkpoint} (step {self.completed_steps})")
        else:
            logger.info("🆕 Starting training from scratch")
            self.completed_steps = 0

    def _load_model_weights(self, checkpoint_path: str):
        """Load model weights from checkpoint (pure PyTorch)"""
        logger.info(f"📥 Loading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle potential module. prefix from DataParallel
        if list(state_dict.keys())[0].startswith("module."):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict, strict=False)
        logger.info("✅ Model weights loaded successfully")

    def _save_checkpoint(self):
        """Save checkpoint (single process)"""
        checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}_pytorch_model.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")

        # Save metadata
        with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps({"steps": self.completed_steps}) + "\n")

        # Save config
        if isinstance(self.config, AccessTrackedConfig):
            output_dir = Path(self.config.output_dir)
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)
            logger.info("✅ Configuration saved")

    def _log_metrics(self, metrics: Dict):
        """Log metrics to console and WandB"""
        if self.completed_steps % self.config.trainer.logging_frequency != 0:
            return

        # Add LR and epoch
        metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
        total_batches = len(self.policy_train_dataloader) if self.policy_train_dataloader else 1
        metrics["epoch"] = self.completed_steps / total_batches

        # Console output
        log_str = f"[Step {self.completed_steps}] "
        log_str += " | ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()])
        logger.info(log_str)

        # WandB
        if wandb.run:
            wandb.log(metrics, step=self.completed_steps)

    def _create_data_iterators(self):
        """Create data iterators"""
        if self.config.framework.name == "QwenGR00T":
            self.policy_iter = iter(self.policy_train_dataloader)
        else:
            self.policy_iter = iter(self.policy_train_dataloader)
            self.fd_iter = iter(self.fd_train_dataloader)
            self.vg_iter = iter(self.vg_train_dataloader)
            self.id_iter = iter(self.id_train_dataloader)

    def _get_next_batch(self):
        """Get next batch with automatic reset"""
        if self.config.framework.name == "QwenGR00T":
            try:
                batch = next(self.policy_iter)
            except StopIteration:
                self.policy_iter = iter(self.policy_train_dataloader)
                batch = next(self.policy_iter)
            return batch, None, None, None

        try:
            batch_policy = next(self.policy_iter)
            batch_fd = next(self.fd_iter)
            batch_vg = next(self.vg_iter)
            batch_id = next(self.id_iter)
        except StopIteration:
            self.policy_iter = iter(self.policy_train_dataloader)
            self.fd_iter = iter(self.fd_train_dataloader)
            self.vg_iter = iter(self.vg_train_dataloader)
            self.id_iter = iter(self.id_train_dataloader)
            batch_policy = next(self.policy_iter)
            batch_fd = next(self.fd_iter)
            batch_vg = next(self.vg_iter)
            batch_id = next(self.id_iter)

        # Assign task types
        for d in batch_policy: d['assigned_task'] = "policy"
        for d in batch_fd: d['assigned_task'] = "forward_dynamics"
        for d in batch_vg: d['assigned_task'] = "video_gen"
        for d in batch_id: d['assigned_task'] = "inverse_dynamics"
        
        return batch_policy + batch_fd + batch_vg + batch_id, None, None, None

    def train(self):
        """Main training loop (pure PyTorch)"""
        self._log_training_config()
        self._create_data_iterators()

        # Progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps),
            desc="Training",
            dynamic_ncols=True
        )

        # Training loop
        while self.completed_steps < self.config.trainer.max_train_steps:
            # Get batch
            t0 = time.time()
            batch_vla, _, _, _ = self._get_next_batch()
            data_time = time.time() - t0

            # Training step
            t0 = time.time()
            step_metrics = self._train_step(batch_vla)
            model_time = time.time() - t0

            # Update progress
            self.completed_steps += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "loss": f"{step_metrics.get('loss', 0):.4f}",
                "data": f"{data_time:.3f}s",
                "model": f"{model_time:.3f}s"
            })

            # Evaluation
            if self.completed_steps % self.config.trainer.eval_interval == 0 and self.completed_steps > 0:
                step_metrics = self.eval_action_model(step_metrics)

            # Add timing metrics
            step_metrics["data_time"] = data_time
            step_metrics["model_time"] = model_time
            self._log_metrics(step_metrics)

            # Save checkpoint
            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()

            # Early stop in debug mode
            if self.config.is_debug and self.completed_steps >= 10:
                logger.warning("🐞 DEBUG MODE: Stopping after 10 steps (set max_train_steps higher to continue)")
                break

        self._finalize_training()
        logger.info("🎉 Training completed successfully!")

    def _train_step(self, batch_vla: list) -> Dict:
        """Single training step (pure PyTorch)"""
        self.model.train()
        
        # Zero gradients every accumulation step
        if self.completed_steps % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Forward pass
        output_dict = self.model.forward(batch_vla)
        total_loss = output_dict["loss"]
        action_loss = output_dict.get("action_loss", total_loss)
        dynamics_loss = output_dict.get("dynamics_loss", None)

        # Backward pass
        (total_loss / self.gradient_accumulation_steps).backward()  # Scale loss for accumulation

        # Optimizer step after accumulation
        if (self.completed_steps + 1) % self.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.trainer.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.trainer.gradient_clipping
                )
            
            # Check for NaN gradients (debug feature)
            if self.config.is_debug:
                for name, param in self.model.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        logger.warning(f"⚠️ NaN gradient detected in {name} at step {self.completed_steps}")
                        if self.config.is_debug:
                            raise ValueError(f"NaN gradient in {name}")

            self.optimizer.step()
            self.lr_scheduler.step()

        # Prepare metrics
        metrics = {
            "loss": total_loss.item(),
            "action_dit_loss": action_loss.item()
        }
        if dynamics_loss is not None:
            metrics["dynamics_loss"] = dynamics_loss.item()
        
        return metrics

    def eval_action_model(self, step_metrics: dict = None) -> dict:
        """Simple evaluation (debug-friendly)"""
        if not self.config.is_debug and self.completed_steps % self.config.trainer.eval_interval != 0:
            return step_metrics or {}

        batch_vla, _, _, _ = self._get_next_batch()
        examples = batch_vla[:min(4, len(batch_vla))]  # Small batch for debug
        
        try:
            self.model.eval()
            with torch.no_grad():
                output_dict = self.model.predict_action(
                    examples=examples,
                    use_ddim=True,
                    num_ddim_steps=5 if self.config.is_debug else 20
                )
            
            normalized_actions = output_dict["normalized_actions"]  # B, T, D
            actions = np.array([ex["action"] for ex in examples])
            mse = np.mean((normalized_actions - actions) ** 2)
            
            step_metrics = step_metrics or {}
            step_metrics["eval_mse"] = float(mse)
            logger.info(f"🔍 Eval MSE: {mse:.6f} (samples={len(examples)})")
            
            self.model.train()
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            if self.config.is_debug:
                raise
        
        return step_metrics or {}

    def _log_training_config(self):
        logger.info("=" * 50)
        logger.info("📋 TRAINING CONFIGURATION")
        logger.info("=" * 50)
        logger.info(f"Device: {self.device}")
        logger.info(f"Max steps: {self.config.trainer.max_train_steps}")
        logger.info(f"Batch size (per device): {self.config.datasets.vla_data.per_device_batch_size}")
        logger.info(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        logger.info(f"Total batch size: {self.total_batch_size}")
        logger.info(f"Learning rate: {self.config.trainer.learning_rate.base}")
        logger.info("=" * 50)

    def _finalize_training(self):
        """Finalize training"""
        # Save final model
        final_path = os.path.join(self.config.output_dir, "final_model", "pytorch_model.pt")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        torch.save(self.model.state_dict(), final_path)
        logger.info(f"✅ Final model saved: {final_path}")

        # Close WandB
        if wandb.run:
            wandb.finish()
            logger.info("⏹️ WandB run finished")


def main(cfg) -> None:
    logger.info("🚀 Starting LDA Single-GPU Debug Trainer (Pure PyTorch)")
    
    # Wrap config
    cfg = wrap_config(cfg)
    
    # Force debug safety settings
    if cfg.is_debug:
        cfg.trainer.gradient_accumulation_steps = 1
        cfg.trainer.logging_frequency = 1
        cfg.trainer.eval_interval = 5
        cfg.trainer.save_interval = 10
        logger.warning("🐞 DEBUG MODE ACTIVE - Using safe defaults for single GPU")

    # Setup directories
    output_dir = setup_directories(cfg)
    
    # Build model
    logger.info("📦 Building model...")
    model = build_framework(cfg)
    logger.info(f"✅ Model built: {model.__class__.__name__}")
    
    # Prepare data
    logger.info("🔍 Preparing data loaders...")
    policy_dl, fd_dl, vg_dl, id_dl = prepare_data(cfg)
    logger.info("✅ Data loaders ready")
    
    # Setup optimizer
    logger.info("⚙️ Setting up optimizer...")
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)
    logger.info("✅ Optimizer ready")
    
    # Create trainer
    trainer = VLATrainer(
        cfg=cfg,
        model=model,
        policy_train_dataloader=policy_dl,
        fd_train_dataloader=fd_dl,
        vg_train_dataloader=vg_dl,
        id_train_dataloader=id_dl,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    
    # Prepare and train
    trainer.prepare_training()
    logger.info("▶️ Starting training loop...")
    trainer.train()
    
    logger.info("🏁 All done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LDA Single-GPU Debug Trainer (Pure PyTorch)")
    parser.add_argument("--config_yaml", type=str, default="LDA/config/training/LDA_cotrain_oxe.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--attach_debugger", action="store_true", help="Wait for debugger attach (port 5678)")
    args, unknown = parser.parse_known_args()

    # Load config
    cfg = OmegaConf.load(args.config_yaml)
    
    # Force debug mode
    if args.debug:
        cfg.is_debug = True
        logger.warning("🐞 --debug flag set: forcing debug mode")
    
    # Apply CLI overrides
    dotlist = normalize_dotlist_args(unknown)
    if dotlist:
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        logger.info(f"🔧 Applied CLI overrides: {dotlist}")

    # Optional debugger attach
    if args.attach_debugger or (cfg.is_debug and getattr(cfg, "attach_debugger", False)):
        import debugpy
        debugpy.listen(("0.0.0.0", 5678))
        logger.warning("⏳ Waiting for debugger attach on port 5678...")
        debugpy.wait_for_client()
        logger.info("🔗 Debugger attached!")

    # Run training with error handling
    try:
        main(cfg)
    except KeyboardInterrupt:
        logger.info("🛑 Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("💥 Training failed with exception:")
        if cfg.is_debug:
            import traceback, pdb
            traceback.print_exc()
            pdb.post_mortem()
        sys.exit(1)