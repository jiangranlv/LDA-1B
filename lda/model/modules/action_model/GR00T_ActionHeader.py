# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with LDA, e.g., "rm "].
# Action repeat is inspired by CogACT



from dataclasses import dataclass, field

import math
import torch.nn.init as init

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from LDA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from LDA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        # self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        # self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))
        self.W = nn.Parameter(torch.empty(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.empty(num_categories, hidden_dim))
        self.init_params()

    def init_params(self):
        # 对每个 category 独立初始化，模仿 nn.Linear
        for i in range(self.num_categories):
            tmp_linear = nn.Linear(self.W.shape[1], self.W.shape[2])  # in_dim -> hidden_dim
            self.W.data[i] = tmp_linear.weight.t().clone()  # 因为 Linear 是 (out, in)，我们存 (in, out)
            self.b.data[i] = tmp_linear.bias.clone()

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


class ActionEncoderWithEmbodiment(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.embodiment_embed = nn.Embedding(num_embodiments, hidden_size)
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)  # [action; time]
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        B, T, _ = actions.shape
        
        # Expand timesteps to (B, T)
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected timesteps shape (B,)")

        # Action embedding
        a_emb = self.layer1(actions)  # (B, T, hidden)

        # Time embedding
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)  # (B, T, hidden)

        # Embodiment embedding: broadcast to all T steps
        e_emb = self.embodiment_embed(cat_ids)  # (B, hidden)
        e_emb = e_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden)


        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))      # (B, T, hidden)
        x = x + e_emb                  # residual-style injection
        x = self.layer3(x)

        return x

class MLPWithEmbodiment(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_embodiments):
        super().__init__()
        self.embodiment_embed = nn.Embedding(num_embodiments, hidden_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        # x: (B, T, input_dim)
        B, T = x.shape[:2]

        # Embodiment embedding
        e_emb = self.embodiment_embed(cat_ids)  # (B, hidden_dim)
        e_emb = e_emb.unsqueeze(1).expand(-1, T, -1)  # (B, T, hidden_dim)

        # Inject embodiment info
        x = F.relu(self.layer1(x))      # (B, T, hidden_dim)
        x = x + e_emb                   # additive conditioning (or use FiLM)
        x = self.layer2(x)              # (B, T, output_dim)

        return x

@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=1, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size # @JinhuiYE
        self.full_config = full_config
        action_model_type = config.action_model_type
        action_model_cfg = DiTConfig[action_model_type]
        
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]
        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.state_dim = config.state_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_timesteps = config.num_inference_timesteps
        self.multi_embodiment = config.max_num_embodiments > 1
        if self.multi_embodiment:
            print("###########################################")
            print(f"Multi Embodiment, using CategorySpecificMLP")
            print("###########################################")
            self.state_encoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,  
                output_dim=self.input_embedding_dim,
                ) if config.state_dim else None
            self.action_encoder = MultiEmbodimentActionEncoder(
                action_dim=config.action_dim,
                hidden_size=self.input_embedding_dim,
                num_embodiments=config.max_num_embodiments,
            )
            self.action_decoder = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=self.hidden_size,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
            # self.state_encoder = MLPWithEmbodiment(
            #     input_dim=config.state_dim,
            #     hidden_dim=self.hidden_size,
            #     output_dim=self.input_embedding_dim,
            #     num_embodiments=config.max_num_embodiments,
            # ) if config.state_dim else None
            # self.action_encoder = ActionEncoderWithEmbodiment(
            #     action_dim=config.action_dim,
            #     hidden_size=self.input_embedding_dim,
            #     num_embodiments=config.max_num_embodiments,
            # )
            # self.action_decoder = MLPWithEmbodiment(
            #     input_dim=self.model.config.output_dim,
            #     hidden_dim=self.hidden_size,
            #     output_dim=self.action_dim,
            #     num_embodiments=config.max_num_embodiments,
            # )
        else:
            print("###########################################")
            print(f"Single Embodiment, using torch MLP")
            print("###########################################")
            self.state_encoder = MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            ) if config.state_dim else None

            self.action_encoder = ActionEncoder(
                action_dim=config.action_dim,
                hidden_size=self.input_embedding_dim,
            )
            self.action_decoder = MLP(
                input_dim=self.model.config.output_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.action_dim,
            )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, 
        state: torch.Tensor = None, encoder_attention_mask=None, embodiment_id=None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise
        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        if self.multi_embodiment:
            action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
            # embed state
            state_features = self.state_encoder(state, embodiment_id) if self.state_dim is not None else None
        else:
            action_features = self.action_encoder(noisy_trajectory, t_discretized)
            state_features = self.state_encoder(state) if self.state_dim is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        # pred = self.action_decoder(model_output)
        # pred_actions = pred[:, -actions.shape[1] :]
        pred = model_output[:, -actions.shape[1] :]
        if self.multi_embodiment:
            pred_actions = self.action_decoder(pred, embodiment_id)
        else:
            pred_actions = self.action_decoder(pred)
        
        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return {"loss": loss}

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None, encoder_attention_mask=None, embodiment_id=None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn( # yes, here make sure action_horizon align with data loader? or share from clinet?
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        if self.multi_embodiment:
            state_features = self.state_encoder(state, embodiment_id) if self.state_dim is not None else None
        else:
            state_features = self.state_encoder(state) if self.state_dim is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            if self.multi_embodiment:
                action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            else:
                action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
                if state_features is not None else torch.cat((future_tokens, action_features), dim=1)


            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
                encoder_attention_mask=encoder_attention_mask,
            )
            # pred = self.action_decoder(model_output)

            # pred_velocity = pred[:, -self.action_horizon :]
            pred = model_output[:, -self.action_horizon :]
            if self.multi_embodiment:
                pred_velocity = self.action_decoder(pred, embodiment_id)
            else:
                pred_velocity = self.action_decoder(pred)

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingActionHead(
        full_config=config
    )

# debug functions
def test_forward_not_equivalent():
    torch.manual_seed(0)

    B, T, Din, Dout = 2, 4, 8, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(B, T, Din, device=device)
    cat_ids = torch.zeros(B, dtype=torch.long, device=device)

    ref = nn.Linear(Din, Dout).to(device)
    test = CategorySpecificLinear(32, Din, Dout).to(device)

    # 强行拷贝权重（注意维度顺序不同）
    test.W.data[0].copy_(ref.weight.data.T)
    test.b.data[0].copy_(ref.bias.data)

    y_ref = ref(x)
    y_test = test(x, cat_ids)

    diff = (y_ref - y_test).abs().max().item()
    print(f"[Forward diff] max |Δ| = {diff:.6e}")
def test_backward_not_equivalent():
    torch.manual_seed(1)

    B, T, Din, Dout = 2, 4, 8, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x1 = torch.randn(B, T, Din, device=device, requires_grad=True)
    x2 = x1.clone().detach().requires_grad_(True)

    cat_ids = torch.ones(B, dtype=torch.long, device=device) * 24

    ref = nn.Linear(Din, Dout).to(device)
    test = CategorySpecificLinear(32, Din, Dout).to(device)

    test.W.data[0].copy_(ref.weight.data.T)
    test.b.data[0].copy_(ref.bias.data)

    ref(x1).sum().backward()
    test(x2, cat_ids).sum().backward()

    w_diff = (ref.weight.grad - test.W.grad[0].T).abs().max().item()
    x_diff = (x1.grad - x2.grad).abs().max().item()

    print(f"[Weight grad diff] max |Δ| = {w_diff:.6e}")
    print(f"[Input grad diff ] max |Δ| = {x_diff:.6e}")
def test_optimizer_divergence():
    torch.manual_seed(7)

    B, T, Din, Dout = 2, 4, 8, 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(B, T, Din, device=device)
    cat_ids = torch.ones(B, dtype=torch.long, device=device) * 24

    ref = nn.Linear(Din, Dout).to(device)
    test = CategorySpecificLinear(32, Din, Dout).to(device)

    test.W.data[0].copy_(ref.weight.data.T)
    test.b.data[0].copy_(ref.bias.data)

    opt_ref = torch.optim.AdamW(ref.parameters(), lr=1e-3)
    opt_test = torch.optim.AdamW(test.parameters(), lr=1e-3)

    for step in range(10):
        opt_ref.zero_grad()
        opt_test.zero_grad()

        ref(x).pow(2).mean().backward()
        test(x, cat_ids).pow(2).mean().backward()

        opt_ref.step()
        opt_test.step()

    w_diff = (ref.weight - test.W[0].T).abs().max().item()
    b_diff = (ref.bias - test.b[0]).abs().max().item()

    print(f"[Weight divergence] max |Δ| = {w_diff:.6e}")
    print(f"[Bias   divergence] max |Δ| = {b_diff:.6e}")

def test_divergence_num_categories_gt_1(
    num_categories=8,
    steps=2000,
    lr=1e-3,
):
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B, T, Din, Dout = 8, 16, 32, 64

    x = torch.randn(B, T, Din, device=device)
    cat_ids = torch.zeros(B, dtype=torch.long, device=device)  # 只用 id=0

    # reference
    ref = nn.Linear(Din, Dout).to(device)

    # test
    test = CategorySpecificLinear(num_categories, Din, Dout).to(device)
    import pdb; pdb.set_trace()
    # 对齐第 0 个 category 的参数
    test.W.data[0].copy_(ref.weight.data.T)
    test.b.data[0].copy_(ref.bias.data)

    opt_ref = torch.optim.AdamW(ref.parameters(), lr=lr, weight_decay=0.01)
    opt_test = torch.optim.AdamW(test.parameters(), lr=lr, weight_decay=0.01)

    divergences = []

    for step in range(steps):
        opt_ref.zero_grad()
        opt_test.zero_grad()

        loss_ref = ref(x).pow(2).mean()
        loss_test = test(x, cat_ids).pow(2).mean()

        loss_ref.backward()
        loss_test.backward()

        opt_ref.step()
        opt_test.step()

        # 每 1000 步记录一次 divergence
        if step % 1000 == 0:
            w_diff = (ref.weight - test.W[0].T).abs().mean().item()
            divergences.append((step, w_diff))
            print(f"[step {step:4d}] mean |ΔW| = {w_diff:.6e}")

    return divergences


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently
    test_divergence_num_categories_gt_1(
        num_categories=1,
        steps=20000,
    )
    # pass
