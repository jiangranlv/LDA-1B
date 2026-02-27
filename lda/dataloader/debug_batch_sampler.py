import argparse
from collections import Counter
import hashlib
import os

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from lda.dataloader import build_multi_task_dataloader
from lda.training.trainer_utils.config_tracker import wrap_config
from lda.training.trainer_utils.trainer_tools import normalize_dotlist_args


def _rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def maybe_init_distributed():
    if not dist.is_available() or dist.is_initialized():
        return False

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"

    dist.init_process_group(backend=backend, init_method="env://")
    print(
        f"[rank={_rank()}/{_world_size()}] initialized distributed backend={backend} local_rank={local_rank}"
    )
    return True


def maybe_cleanup_distributed(was_initialized_here: bool):
    if not was_initialized_here:
        return
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def summarize_batch(batch, batch_idx: int):
    task_counter = Counter()
    embodiment_counter = Counter()

    for sample in batch:
        task_counter[sample.get("assigned_task", "missing")] += 1
        embodiment_counter[sample.get("embodiment_id", "missing")] += 1

    fingerprints = [sample_fingerprint(sample) for sample in batch]
    unique_fingerprints = len(set(fingerprints))

    print(f"[rank={_rank()}/{_world_size()}] batch={batch_idx} size={len(batch)}")
    print(f"  tasks: {dict(task_counter)}")
    print(f"  embodiment_ids(top): {dict(embodiment_counter)}")
    print(f"  fingerprints(unique): {unique_fingerprints}/{len(fingerprints)}")
    print(f"  fingerprints(head): {fingerprints[: min(8, len(fingerprints))]}")


def _short_text(x, max_len: int = 48) -> str:
    if x is None:
        return ""
    s = str(x).replace("\n", " ").strip()
    return s[:max_len]


def sample_fingerprint(sample) -> str:
    task = str(sample.get("assigned_task", "na"))
    embodiment = str(sample.get("embodiment_id", "na"))
    action = sample.get("action", None)

    if action is None:
        fallback = _short_text(sample.get("lang", ""))
        digest = hashlib.sha1(fallback.encode("utf-8")).hexdigest()[:10]
        return f"{task}:{embodiment}:lang:{digest}"

    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy()
    else:
        action_np = np.asarray(action)

    # Round to reduce noise from tiny float differences when comparing across ranks.
    action_np = np.round(action_np.astype(np.float32, copy=False), 4)
    digest = hashlib.sha1(action_np.tobytes()).hexdigest()[:10]
    return f"{task}:{embodiment}:act:{digest}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/mnt/home/liukai/code/LDA/lda/config/training/LDA_pretrain.yaml",
        help="Path to YAML config",
    )
    parser.add_argument("--num_batches", type=int, default=3, help="How many batches to inspect")
    args, clipargs = parser.parse_known_args()
    dist_inited_here = maybe_init_distributed()

    try:
        cfg = OmegaConf.load(args.config_yaml)
        dotlist = normalize_dotlist_args(clipargs)
        cli_cfg = OmegaConf.from_dotlist(dotlist)
        cfg = OmegaConf.merge(cfg, cli_cfg)
        cfg = wrap_config(cfg)
        cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
        dataloader = build_multi_task_dataloader(
            cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py
        )

        print(
            f"[rank={_rank()}/{_world_size()}] dataloader_len={len(dataloader)} "
            f"batch_sampler={type(dataloader.batch_sampler).__name__}"
        )

        it = iter(dataloader)
        for i in range(args.num_batches):
            batch = next(it)
            breakpoint()
            summarize_batch(batch, i)
    finally:
        maybe_cleanup_distributed(dist_inited_here)


if __name__ == "__main__":
    main()
