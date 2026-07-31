"""BFN for DRBFN-QVPO: outputs per-agent potential Φ ∈ ℝ^N.

Key changes from v_final BFN (utils/drbfn.py):
- Output dim: N (per-agent potential), NOT N-1 (stick-breaking)
- No _w_to_r: r is computed externally via PBRS formula
- Supports log_prob computation for Q-weighted VLB training
- Action-aware: input is (s, a_joint)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D


def safe_exp(x):
    return x.clamp(min=-10, max=10).exp()


class CtsBayesianFlow(nn.Module):
    """Continuous Bayesian Flow: forward noise process for BFN (unchanged)."""

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
    """Conditional denoising network: predicts (mean, logvar) from noisy Φ."""

    def __init__(self, cond_dim, output_dim, hidden=64):
        super().__init__()
        self.output_dim = output_dim
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(output_dim + hidden + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * output_dim),  # (mean, logvar)
        )

    def forward(self, x_t, t, h):
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x_t, t_emb, h], dim=-1))


class PotentialBFN(nn.Module):
    """BFN that outputs per-agent potential Φ ∈ ℝ^N (no stick-breaking).

    Input: (s, a_joint) → encoded h
    Output: Φ = (Φ_1, ..., Φ_N) ∈ ℝ^N
    """

    def __init__(self, raw_dim, n_agents, hidden=64,
                 min_variance=1e-3, n_sample_steps=2):
        super().__init__()
        self.n_agents = n_agents
        self.hidden = hidden
        self.n_sample_steps = n_sample_steps
        self.D = n_agents  # output dim = N (per-agent potential)

        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow = CtsBayesianFlow(min_variance=min_variance)
        self.g_net = ConditionNet(cond_dim=hidden, output_dim=self.D, hidden=hidden)

    def encode(self, raw):
        return self.encoder(raw)

    @torch.no_grad()
    def sample_phi(self, h):
        """BFN reverse sampling: prior → Φ. h: [B, hidden] → Φ: [B, N]."""
        B, device = h.shape[0], h.device
        D = self.D
        mean, precision = self.flow.get_prior_input_params(B * D, device)
        mean = mean.reshape(B, D)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean, p_logvar = out[:, :D], out[:, D:]
            # Tight clamp: logvar in [-2, 2] → std in [0.37, 2.7], prevents sampling explosion
            p_logvar = p_logvar.clamp(-2, 2)
            p_std = safe_exp(p_logvar * 0.5)
            output_sample = p_mean + p_std * torch.randn_like(p_mean)
            alpha = self.flow.get_alpha(i, self.n_sample_steps)
            y = self.flow.get_sender_dist(output_sample, alpha).sample()
            new_mean, new_prec = self.flow.update_input_params(
                (mean.reshape(-1, 1), precision), y.reshape(-1, 1), alpha)
            mean = new_mean.reshape(B, D)
            precision = new_prec

        t_final = torch.ones(B, 1, device=device)
        return self.g_net(mean, t_final, h)[:, :D]

    def log_prob_phi(self, phi, h):
        """log p_BFN(Φ | s, a) using fixed-variance Gaussian at final step.

        Matches original BFN reconstruction_loss logic (line 225-244 of
        bayesian-flow-networks/model.py):
            noise_dev = sqrt(min_variance)
            final_dist = Normal(mean, noise_dev)
            log_p = -final_dist.log_prob(data)

        Using FIXED variance (not network-predicted) prevents σ collapse
        and aligns with BFN theory.

        Args:
            phi: [B, N] the sample (treated as data)
            h: [B, hidden] condition

        Returns:
            log_prob: [B] log-probability of phi under BFN
        """
        B = phi.shape[0]
        device = phi.device
        D = self.D

        # Forward BFN to final step (deterministic version: use predicted mean as y)
        mean, precision = self.flow.get_prior_input_params(B * D, device)
        mean = mean.reshape(B, D)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean = out[:, :D]

            # Use predicted mean as deterministic "input update"
            alpha = self.flow.get_alpha(i, self.n_sample_steps)
            y = p_mean
            new_mean, new_prec = self.flow.update_input_params(
                (mean.reshape(-1, 1), precision), y.reshape(-1, 1), alpha)
            mean = new_mean.reshape(B, D)
            precision = new_prec

        # Final prediction
        t_final = torch.ones(B, 1, device=device)
        out_final = self.g_net(mean, t_final, h)
        pred_mean = out_final[:, :D]

        # Fixed noise_dev (matches original BFN reconstruction_loss)
        noise_dev = math.sqrt(self.flow.min_variance)
        var = noise_dev ** 2

        log_p = -0.5 * ((phi - pred_mean) ** 2 / var + math.log(2 * math.pi * var))
        return log_p.sum(dim=-1)  # [B], sum over N dims

    @torch.no_grad()
    def sample(self, raw):
        """Full sampling: raw → encode → BFN sample → Φ.
        raw: [B, raw_dim] → Φ: [B, N]."""
        h = self.encode(raw)
        return self.sample_phi(h)

    def train_bfn(self, h, phi_target):
        """Legacy training (MSE on noisy target) — kept for compatibility.
        For Q-weighted VLB training, use log_prob_phi directly.
        """
        B = phi_target.shape[0]
        device = phi_target.device
        t = torch.rand(B, 1, device=device) * 0.8 + 0.1
        x_t = self.flow(phi_target, t)
        out = self.g_net(x_t, t, h)
        pred_mean, pred_logvar = out[:, :self.D], out[:, self.D:]
        pred_logvar = pred_logvar.clamp(-5, 5)

        loss_d = (pred_mean - phi_target).pow(2).mean()
        return loss_d
