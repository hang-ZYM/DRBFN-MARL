"""
DRBFN: Bayesian Flow Network for multi-agent credit assignment.

Decomposes global reward R into individual rewards r_i (sum = R).
Uses BFN generative process with Bellman residuals as training signal.
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
    """Conditional denoising network for BFN: predicts (mean, logvar) from noisy input."""

    def __init__(self, cond_dim, n_agents=2, hidden=64):
        super().__init__()
        D = n_agents - 1
        self.out_dim = 2 * D
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(D + hidden + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.out_dim),
        )

    def forward(self, x_t, t, h):
        t_emb = self.t_embed(t)
        return self.net(torch.cat([x_t, t_emb, h], dim=-1))


class UnifiedWorldModel(nn.Module):
    """
    DRBFN: shared encoder phi + BFN reward generation head G.

    Input:  [state; one_hot(a_1); ...; one_hot(a_N)]
    Output: (r_1, ..., r_N) where r_i >= 0 and sum(r_i) = R
    """

    def __init__(self, raw_dim, n_actions=3, n_agents=2, hidden=64,
                 min_variance=1e-3, n_sample_steps=2, n_candidates=8):
        super().__init__()
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden = hidden
        self.n_sample_steps = n_sample_steps
        self.D = n_agents - 1

        self.encoder = nn.Sequential(
            nn.Linear(raw_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.flow = CtsBayesianFlow(min_variance=min_variance)
        self.g_net = ConditionNet(cond_dim=hidden, n_agents=n_agents, hidden=hidden)

    def encode(self, raw):
        """raw: [B, raw_dim] -> h: [B, hidden]"""
        return self.encoder(raw)

    def _w_to_r(self, w, R):
        """Convert BFN weights to individual rewards. Returns (r_1,...,r_N, w_01)."""
        B = w.shape[0]
        R = R.view(B, 1)
        N = self.n_agents

        if N == 2:
            w01 = ((w[:, :1] + 1.0) / 2.0).clamp(0.0, 1.0)
            w_all = torch.cat([w01, 1.0 - w01], dim=-1)
        else:
            w_clamped = ((w + 1.0) / 2.0).clamp(0.0, 1.0)
            remaining = torch.ones(B, 1, device=w.device)
            weights = []
            for d in range(N - 1):
                w_d = w_clamped[:, d:d+1] * remaining
                weights.append(w_d)
                remaining = remaining - w_d
            weights.append(remaining)
            w_all = torch.cat(weights, dim=-1)

        r = w_all * R
        result = [r[:, i] for i in range(N)]
        result.append(w_all[:, 0])
        return tuple(result)

    @torch.no_grad()
    def sample_w(self, h):
        """BFN reverse sampling: prior -> denoised weights. h: [B, hidden] -> w: [B, D]."""
        B, device = h.shape[0], h.device
        D = self.D
        mean, precision = self.flow.get_prior_input_params(B * D, device)
        mean = mean.reshape(B, D)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean, p_logvar = out[:, :D], out[:, D:]
            # [同步修复] clamp logvar, 与 train_bfn 保持一致, 防止采样时 std 爆炸
            p_logvar = p_logvar.clamp(-5, 5)
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

    @torch.no_grad()
    def predict_w(self, h):
        """Deterministic BFN inference (probability flow ODE).
        Uses predicted mean at each step instead of sampling.
        h: [B, hidden] -> (w_mean: [B, D], w_var: [B, D])."""
        B, device = h.shape[0], h.device
        D = self.D
        mean, precision = self.flow.get_prior_input_params(B * D, device)
        mean = mean.reshape(B, D)

        for i in range(1, self.n_sample_steps + 1):
            t = torch.ones(B, 1, device=device) * (i - 1) / self.n_sample_steps
            out = self.g_net(mean, t, h)
            p_mean = out[:, :D]
            # Deterministic: use predicted mean as observation directly
            alpha = self.flow.get_alpha(i, self.n_sample_steps)
            y = p_mean
            new_mean, new_prec = self.flow.update_input_params(
                (mean.reshape(-1, 1), precision), y.reshape(-1, 1), alpha)
            mean = new_mean.reshape(B, D)
            precision = new_prec

        t_final = torch.ones(B, 1, device=device)
        out_final = self.g_net(mean, t_final, h)
        w_mean = out_final[:, :D]
        w_logvar = out_final[:, D:].clamp(-5, 5)
        return w_mean, safe_exp(w_logvar)

    @torch.no_grad()
    def sample(self, raw, R):
        """Full inference: raw -> encode -> BFN sample -> individual rewards.
        raw: [B, raw_dim], R: [B] -> tuple of (r_1,...,r_N, w_01)."""
        h = self.encode(raw)
        return self._w_to_r(self.sample_w(h), R)

    @torch.no_grad()
    def predict(self, raw, R):
        """Deterministic inference: raw -> encode -> BFN predict -> rewards + variance.
        raw: [B, raw_dim], R: [B] -> (r_tuple, w_var: [B, D])."""
        h = self.encode(raw)
        w_mean, w_var = self.predict_w(h)
        r_tuple = self._w_to_r(w_mean, R)
        return r_tuple, w_var

    def train_bfn(self, h, w_target):
        """BFN training: noise target -> denoise -> loss.
        h: [B, hidden], w_target: [B, D] -> (loss_d, loss_e)."""
        B = w_target.shape[0]
        device = w_target.device
        t = torch.rand(B, 1, device=device) * 0.8 + 0.1
        x_t = self.flow(w_target, t)
        out = self.g_net(x_t, t, h)
        D = self.D
        pred_mean, pred_logvar = out[:, :D], out[:, D:]

        loss_d = (pred_mean - w_target).pow(2).mean()

        # [修复] 熵正则项 loss_e 会无上界推大 logvar 导致训练爆炸
        # 原始: loss_e = -pred_logvar.mean()
        # 原因: logvar 无上界, 优化器会无限增大 logvar 来最小化 loss_e
        # 修复: clamp logvar 到 [-5, 5], 对应 σ² ∈ [e^{-5}, e^{5}] ≈ [0.007, 148]
        #       既保留探索性(方差>0), 又防止数值爆炸
        pred_logvar_clamped = pred_logvar.clamp(-5, 5)
        loss_e = -pred_logvar_clamped.mean()

        return loss_d, loss_e
