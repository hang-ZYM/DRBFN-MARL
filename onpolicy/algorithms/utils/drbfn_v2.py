"""DRBFN-v2 BFN: per-agent reward generator (generative paradigm).

Key changes from v1 (utils/drbfn.py):
  1. Output is r_i ∈ ℝ^N directly, NOT stick-breaking weights.
     Conservation (Σr_i = R) becomes a soft prior (added as loss term),
     not a hard structural constraint. This leaves room for synergy effects.
  2. Multi-sample sampling API: sample_k() returns K samples for variance
     estimation and data augmentation.
  3. Conditioning includes R as a feature (so generator can scale r_i with R).

Paradigm: per-agent reward *generation* (not decomposition).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def safe_exp(data):
    return data.clamp(min=-10, max=10).exp()


class CtsBayesianFlow(nn.Module):
    """Continuous Bayesian Flow: forward noise process for BFN."""

    def __init__(self, min_variance=1e-3):
        super().__init__()
        self.min_variance = min_variance

    @torch.no_grad()
    def forward(self, x_0, t):
        post_var = torch.pow(self.min_variance, t)
        alpha_t = 1 - post_var
        mean_mean = alpha_t * x_0
        mean_var = alpha_t * post_var
        noise = torch.randn_like(mean_mean)
        return mean_mean + mean_var.sqrt() * noise

    def get_prior_input_params(self, batch_size, device):
        return torch.zeros(batch_size, 1, device=device), 1.0

    def get_alpha(self, i, n_steps):
        sigma_1 = math.sqrt(self.min_variance)
        return (sigma_1 ** (-2 * i / n_steps)) * (1 - sigma_1 ** (2 / n_steps))

    def update_input_params(self, params, y, alpha):
        mean, precision = params
        new_precision = precision + alpha
        new_mean = ((precision * mean) + (alpha * y)) / new_precision
        return new_mean, new_precision

    def get_sender_dist(self, x, alpha):
        return D.Normal(x, 1.0 / alpha ** 0.5)


class ConditionNet(nn.Module):
    """Conditional denoising network: predicts (mean, logvar) for r ∈ ℝ^N.

    Output dim is 2*N (N for mean, N for logvar), instead of v1's 2*(N-1) weights.
    """

    def __init__(self, cond_dim, n_agents=2, hidden=64):
        super().__init__()
        self.N = n_agents
        self.out_dim = 2 * n_agents
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(n_agents + hidden + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x_t, t, h):
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x_t, t_emb, h], dim=-1))


class RewardGenerator(nn.Module):
    """DRBFN-v2 BFN: generates per-agent reward vector r ∈ ℝ^N.

    Input conditioning:  [state; one_hot(a_1); ...; one_hot(a_N); R]
    Output:              (r_1, ..., r_N) directly (no stick-breaking).

    Conservation is enforced softly via loss term, not structurally.
    """

    def __init__(self, raw_dim, n_actions=3, n_agents=2, hidden=64,
                 min_variance=1e-3, n_sample_steps=2):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden = hidden
        self.n_sample_steps = n_sample_steps
        self.N = n_agents

        # Encoder: condition on (state, joint_action, R) -> h
        # raw_dim already includes the R dimension (passed by caller)
        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow = CtsBayesianFlow(min_variance=min_variance)
        # Denoising net: predict (mean, logvar) for r ∈ ℝ^N
        self.g_net = ConditionNet(cond_dim=hidden, n_agents=n_agents, hidden=hidden)

    def encode(self, raw):
        """raw: [B, raw_dim] -> h: [B, hidden]"""
        return self.encoder(raw)

    @torch.no_grad()
    def sample_r(self, h):
        """Stochastic BFN reverse sampling. h: [B, hidden] -> r: [B, N]."""
        B, device = h.shape[0], h.device
        N = self.N
        mean, precision = self.flow.get_prior_input_params(B * N, device)
        mean = mean.reshape(B, N)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean, p_logvar = out[:, :N], out[:, N:]
            p_logvar = p_logvar.clamp(-5, 5)
            p_std = safe_exp(p_logvar * 0.5)
            output_sample = p_mean + p_std * torch.randn_like(p_mean)
            alpha = self.flow.get_alpha(i, self.n_sample_steps)
            y = self.flow.get_sender_dist(output_sample, alpha).sample()
            new_mean, new_prec = self.flow.update_input_params(
                (mean.reshape(-1, 1), precision), y.reshape(-1, 1), alpha)
            mean = new_mean.reshape(B, N)
            precision = new_prec

        t_final = torch.ones(B, 1, device=device)
        return self.g_net(mean, t_final, h)[:, :N]

    @torch.no_grad()
    def predict_r(self, h):
        """Deterministic BFN inference (probability flow ODE).
        h: [B, hidden] -> (r_mean: [B, N], r_var: [B, N])."""
        B, device = h.shape[0], h.device
        N = self.N
        mean, precision = self.flow.get_prior_input_params(B * N, device)
        mean = mean.reshape(B, N)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean = out[:, :N]
            alpha = self.flow.get_alpha(i, self.n_sample_steps)
            y = p_mean
            new_mean, new_prec = self.flow.update_input_params(
                (mean.reshape(-1, 1), precision), y.reshape(-1, 1), alpha)
            mean = new_mean.reshape(B, N)
            precision = new_prec

        t_final = torch.ones(B, 1, device=device)
        out_final = self.g_net(mean, t_final, h)
        r_mean = out_final[:, :N]
        r_logvar = out_final[:, N:].clamp(-5, 5)
        return r_mean, safe_exp(r_logvar)

    @torch.no_grad()
    def sample_k(self, raw, K=4):
        """Multi-sample API: draw K samples of r ∈ ℝ^N.
        raw: [B, raw_dim] -> r_samples: [B, K, N]."""
        h = self.encode(raw)
        samples = [self.sample_r(h) for _ in range(K)]
        return torch.stack(samples, dim=1)  # [B, K, N]

    @torch.no_grad()
    def predict(self, raw):
        """Deterministic: raw -> (r_mean, r_var)."""
        h = self.encode(raw)
        return self.predict_r(h)

    def train_bfn(self, h, r_target):
        """BFN training: noise target -> denoise -> loss.
        h: [B, hidden], r_target: [B, N] -> (loss_d, loss_e)."""
        B = r_target.shape[0]
        device = r_target.device
        N = self.N
        t = torch.rand(B, 1, device=device) * 0.8 + 0.1
        x_t = self.flow(r_target, t)
        out = self.g_net(x_t, t, h)
        pred_mean, pred_logvar = out[:, :N], out[:, N:]

        loss_d = (pred_mean - r_target).pow(2).mean()

        # Entropy regularization (same fix as v1: clamp logvar)
        pred_logvar_clamped = pred_logvar.clamp(-5, 5)
        loss_e = -pred_logvar_clamped.mean()

        return loss_d, loss_e
