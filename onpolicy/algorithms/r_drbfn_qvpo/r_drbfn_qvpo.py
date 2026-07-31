"""R_DRBFN_QVPO Trainer: BFN outputs Φ, r via PBRS, Q-weighted VLB training.

Key design (from DRBFN_思路.md):
  - BFN(s, a) → Φ ∈ ℝ^N  (per-agent potential)
  - r_i = R/N + γΦ_i(s', a') - Φ_i(s, a)  (PBRS form, action-aware, SARSA)
  - Actor trained with r_i as per-agent reward (standard PPO, V-critic baseline)
  - Q_tot trained on R via n-step return (independent of BFN)
  - BFN trained via Q-weighted VLB:
      g_i = Q_tot(s,a) - Q_tot(s, default_i, a_{-i})  (default action counterfactual)
      sample K Φ candidates, each gives r^(k)
      align^(k) = Σ_i r_i^(k) · g_i
      weights^(k) = ReLU(normalize(align))
      L_BFN = -Σ_k weights · log p_BFN(Φ^(k) | s, a)
  - Warmup: first N steps use R/N as r (no BFN), to let Q_tot learn first
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check


class R_DRBFN_QVPO(R_MAPPO):
    """Trainer for DRBFN-QVPO."""

    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.train_step_count = 0

        self.n_step = getattr(args, 'drbfn_n_step', 5)
        self.max_grad_norm = args.max_grad_norm

        self._stats = {
            'drbfn_loss': 0.0,
            'qtot_loss': 0.0,
            'g_i_mean': 0.0,
            'g_i_std': 0.0,
            'align_mean': 0.0,
            'bfn_sigma_mean': 0.0,
            'r_deviation_from_RN': 0.0,
            'n_step_return_mean': 0.0,
            'in_warmup': 0.0,
        }

    @property
    def p(self):
        return self.policy

    def train(self, buffer, update_actor=True):
        total_steps = self.train_step_count * buffer.episode_length * buffer.n_rollout_threads
        in_warmup = total_steps < self.p.warmup_t
        self._stats['in_warmup'] = float(in_warmup)

        # Save original buffer rewards/returns to restore after PPO
        original_rewards = buffer.rewards.copy()
        original_returns = buffer.returns.copy()

        # Compute per-agent r via BFN (or R/N during warmup)
        if in_warmup:
            per_agent_rewards = self._compute_warmup_rewards(buffer)
        else:
            per_agent_rewards, gen_stats = self._compute_pbrs_rewards(buffer)
            self._stats.update(gen_stats)

        # Replace buffer rewards with per-agent r_i (broadcast across action dim)
        # buffer.rewards shape: (T+1, env, N, 1)
        buffer.rewards[:] = per_agent_rewards

        value_normalizer = self.value_normalizer
        next_value = buffer.value_preds[-1].copy()
        buffer.compute_returns(next_value, value_normalizer)

        # Standard PPO training (uses per-agent r_i rewards for advantage)
        train_info = super().train(buffer, update_actor)

        # Restore buffer
        buffer.rewards[:] = original_rewards
        buffer.returns[:] = original_returns
        buffer.compute_returns(next_value, value_normalizer)

        # Train components: Q_tot + BFN
        if in_warmup:
            self._train_warmup_qtot(buffer)
        else:
            comp_info = self._train_components(buffer)
            train_info.update(comp_info)

        self.train_step_count += 1
        return train_info

    # ====================================================================
    # Warmup: use R/N as per-agent reward
    # ====================================================================
    def _compute_warmup_rewards(self, buffer):
        T = buffer.episode_length
        R_np = buffer.rewards[:T, :, 0, :]  # (T, env, 1)
        N = self.num_agents
        # broadcast R/N to each agent
        per_agent = np.broadcast_to(R_np / N, (T, R_np.shape[1], N))
        # Shape (T, env, N, 1) — matches buffer.rewards
        result = np.zeros((T, R_np.shape[1], N, 1), dtype=np.float32)
        result[:, :, :, 0] = per_agent
        return result

    # ====================================================================
    # Compute per-agent r via PBRS form (K-argmax deployment)
    # ====================================================================
    def _compute_pbrs_rewards(self, buffer):
        """For each (s_t, a_t), compute r_i = R/N + γΦ_i(s_{t+1}, a_{t+1}) - Φ_i(s_t, a_t).
        Use K-sample argmax to pick best Φ.
        """
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length

        share_obs_np = buffer.share_obs[:T, :, 0, :]  # (T, env, share_obs_dim)
        actions_np = buffer.actions[:T]  # (T, env, N, 1)
        rewards_np = buffer.rewards[:T, :, 0, :]  # (T, env, 1)
        share_obs_next_np = buffer.share_obs[1:T+1, :, 0, :]  # (T, env, share_obs_dim)
        actions_next_np = np.concatenate([
            buffer.actions[1:T],
            np.zeros_like(buffer.actions[:1])  # placeholder for terminal step
        ], axis=0)
        masks_next_np = buffer.masks[1:T+1, :, 0, :]  # (T, env, 1), 0=done

        num_actions = self.p.num_actions
        T_env = T * share_obs_np.shape[1]

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)  # (T, env, 1)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        actions_next = check(actions_next_np).to(**tpdv).long()
        masks_next = check(masks_next_np).to(**tpdv)  # (T, env, 1)

        # Flatten actions: (T, env, N) → joint action one-hot
        actions_sq = actions.squeeze(-1)  # (T, env, N)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T_env, N * num_actions)

        actions_next_sq = actions_next.squeeze(-1)
        actions_next_oh = F.one_hot(actions_next_sq, num_actions).float()
        joint_a_next = actions_next_oh.reshape(T_env, N * num_actions)

        so_flat = share_obs.reshape(T_env, -1)
        so_next_flat = share_obs_next.reshape(T_env, -1)
        R_flat = R.reshape(T_env)
        # Flatten masks_next from (T, env, 1) → (T_env, 1)
        mask_next_flat = masks_next.reshape(T_env, 1)

        drbfn_input = torch.cat([so_flat, joint_a], dim=-1)
        drbfn_input_next = torch.cat([so_next_flat, joint_a_next], dim=-1)

        # Encode h once
        with torch.no_grad():
            h = self.p.drbfn.encode(drbfn_input)
            h_next = self.p.drbfn.encode(drbfn_input_next)

            # Compute g_i (for K-argmax selection)
            g_i = self._compute_g_i(so_flat, joint_a)  # (T_env, N)

            # K-sample argmax
            K = self.p.K_deploy
            best_phi = None
            best_align = None

            for k in range(K):
                phi_k = self.p.drbfn.sample_phi(h)
                phi_k = phi_k.clamp(-self.p.phi_clamp, self.p.phi_clamp)
                phi_next_k = self.p.drbfn.sample_phi(h_next)
                phi_next_k = phi_next_k.clamp(-self.p.phi_clamp, self.p.phi_clamp)

                # r_i = R/N + γΦ_i(s', a') - Φ_i(s, a)
                # mask_next_flat: 0 at terminal → no future Φ term
                r_k = (R_flat / N).unsqueeze(-1) + self.gamma * phi_next_k * mask_next_flat - phi_k
                # align = r · g
                align_k = (r_k * g_i).sum(dim=-1)  # (T_env,)

                if best_align is None or align_k.mean().item() > best_align.mean().item():
                    best_align = align_k
                    best_phi = phi_k
                    best_phi_next = phi_next_k

            # Compute final r with best Φ
            r_final = (R_flat / N).unsqueeze(-1) + self.gamma * best_phi_next * mask_next_flat - best_phi
            # zero out terminal-step shaping (mask=0 → r_i = R/N - Φ_i)
            # Actually for terminal (mask=0): r_i = R/N - Φ_i, which is the PBRS terminal form
            # For non-terminal (mask=1): r_i = R/N + γΦ' - Φ

        r_np = r_final.cpu().numpy()
        r_np = r_np.reshape(T, -1, N, 1)

        # Shape (T, env, N, 1) — matches buffer.rewards
        result = np.zeros((T, r_np.shape[1], N, 1), dtype=np.float32)
        result[:] = r_np

        # Stats
        R_mean = R_flat.mean().item()
        r_mean = r_final.mean().item()
        deviation = (r_final - (R_flat / N).unsqueeze(-1)).abs().mean().item()

        stats = {
            'g_i_mean': g_i.mean().item(),
            'g_i_std': g_i.std().item() if g_i.numel() > 1 else 0.0,
            'align_mean': best_align.mean().item(),
            'r_deviation_from_RN': deviation,
        }

        return result, stats

    # ====================================================================
    # Compute g_i = Q_tot(s, a) - Q_tot(s, default_i, a_{-i}) per agent
    # ====================================================================
    def _compute_g_i(self, so_flat, joint_a_flat):
        """Default-action counterfactual for per-agent Q sensitivity.

        Args:
            so_flat: (B, share_obs_dim)
            joint_a_flat: (B, N * num_actions)

        Returns:
            g_i: (B, N) per-agent Q_tot sensitivity
        """
        N = self.num_agents
        num_actions = self.p.num_actions
        default_action = self.p.default_action

        # Q_tot at actual action
        q_actual = self.p.qtot_net(so_flat, joint_a_flat)  # (B, 1)

        # For each agent, build "default" joint action (replace a_i with default)
        batch_size = so_flat.shape[0]
        joint_a = joint_a_flat.reshape(batch_size, N, num_actions)

        g_i_list = []
        for i in range(N):
            a_default = joint_a.clone()
            # zero out agent i's current action
            a_default[:, i, :] = 0
            # set default action to 1
            a_default[:, i, default_action] = 1
            joint_a_def = a_default.reshape(batch_size, N * num_actions)
            q_default_i = self.p.qtot_net(so_flat, joint_a_def)  # (B, 1)
            g_i_list.append((q_actual - q_default_i).squeeze(-1))  # (B,)

        g_i = torch.stack(g_i_list, dim=-1)  # (B, N)
        return g_i

    # ====================================================================
    # Q_tot training: n-step return on R
    # ====================================================================
    def _train_qtot(self, buffer):
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        share_obs_np = buffer.share_obs[:T, :, 0, :]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        masks_np = buffer.masks[:T]
        next_masks_np = buffer.masks[1:T+1, :, 0, :]
        share_obs_full_np = buffer.share_obs[:T+1, :, 0, :]

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)  # (T, env, 1)
        next_masks = check(next_masks_np).to(**tpdv)
        share_obs_full = check(share_obs_full_np).to(**tpdv)

        T_env = T * share_obs.shape[1]
        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, -1, N * num_actions)

        # n-step return G_n = Σ γ^k R_{t+k} + γ^n Q_target(s_{t+n}, a_{t+n})
        n = self.n_step
        gamma = self.gamma

        G_n = torch.zeros_like(R)
        rewards_full = R

        # Sum of discounted rewards over next n steps
        for k in range(n):
            if k < T:
                if k == 0:
                    r_k = rewards_full
                else:
                    r_k = torch.zeros_like(rewards_full)
                    r_k[:T-k] = rewards_full[k:T]
                G_n = G_n + (gamma ** k) * r_k

        # term_cum: cumulative terminated flag within window (zero out bootstrap)
        term_cum = torch.zeros_like(R)
        for k in range(n):
            if k == 0:
                term_cum = term_cum + next_masks  # actually we want "terminated"
            else:
                t_shifted = torch.zeros_like(R)
                if k < T:
                    t_shifted[:T-k] = next_masks[k:T]
                term_cum = term_cum + t_shifted
        terminated_full = 1.0 - next_masks  # terminated at next step
        # Cumulative terminated: 1 if any step in window is terminated
        term_cum = torch.zeros_like(R)
        for k in range(n):
            if k < T:
                term_shifted = torch.zeros_like(R)
                term_shifted[:T-k] = terminated_full[k:T]
                term_cum = term_cum + term_shifted
        valid_bootstrap = (1.0 - term_cum.clamp(max=1.0))

        # Bootstrap: γ^n Q_target(s_{t+n}, a_{t+n})
        # For t+n > T, no bootstrap
        Q_next = torch.zeros_like(R)
        if n < T:
            so_next = share_obs_full[n:T+1]  # (T-n+1, env, dim) — careful with bounds
            # Actually want s_{t+n} for t in [0, T-n-1], so index [n, T)
            so_next = share_obs_full[n:T]
            ja_next = joint_a[n:T]
            q_next_val = self.p.qtot_target(so_next, ja_next)
            Q_next[:T-n] = q_next_val

        bootstrap_factor = gamma ** n
        G_n = G_n + bootstrap_factor * Q_next * valid_bootstrap

        # Q_tot prediction
        q_tot_pred = self.p.qtot_net(share_obs, joint_a)  # (T, env, 1)
        qtot_loss = ((q_tot_pred - G_n.detach()) ** 2).mean()

        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qtot_net.parameters(), self.max_grad_norm)
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()

        return qtot_loss.item(), G_n.mean().item()

    # ====================================================================
    # Warmup Q_tot training: same as above (only Q_tot, no BFN)
    # ====================================================================
    def _train_warmup_qtot(self, buffer):
        return self._train_qtot(buffer)

    # ====================================================================
    # BFN training: Q-weighted VLB on K samples
    # ====================================================================
    def _train_bfn(self, buffer):
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        share_obs_np = buffer.share_obs[:T, :, 0, :]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        share_obs_next_np = buffer.share_obs[1:T+1, :, 0, :]
        actions_next_np = np.concatenate([
            buffer.actions[1:T],
            np.zeros_like(buffer.actions[:1])
        ], axis=0)
        masks_next_np = buffer.masks[1:T+1, :, 0, :]

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        actions_next = check(actions_next_np).to(**tpdv).long()
        masks_next = check(masks_next_np).to(**tpdv)

        T_env = T * share_obs.shape[1]
        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T_env, N * num_actions)
        so_flat = share_obs.reshape(T_env, -1)
        R_flat = R.reshape(T_env)
        # Flatten masks_next from (T, env, 1) → (T_env, 1)
        mask_next_flat = masks_next.reshape(T_env, 1)

        actions_next_sq = actions_next.squeeze(-1)
        actions_next_oh = F.one_hot(actions_next_sq, num_actions).float()
        joint_a_next = actions_next_oh.reshape(T_env, N * num_actions)
        so_next_flat = share_obs_next.reshape(T_env, -1)

        drbfn_input = torch.cat([so_flat, joint_a], dim=-1)
        drbfn_input_next = torch.cat([so_next_flat, joint_a_next], dim=-1)

        # Compute g_i (detached — used as weight only)
        with torch.no_grad():
            g_i = self._compute_g_i(so_flat, joint_a)  # (T_env, N)
            h_next = self.p.drbfn.encode(drbfn_input_next)

        # Compute h (require grad for BFN training)
        h = self.p.drbfn.encode(drbfn_input)

        K = self.p.K_train
        aligns_list = []
        log_ps_list = []
        sigmas_list = []

        for k in range(K):
            # Sample Φ (needs grad for log_prob)
            phi = self.p.drbfn.sample_phi(h)
            phi = phi.clamp(-self.p.phi_clamp, self.p.phi_clamp)

            # Sample Φ_next (detached, used only to compute r for align)
            with torch.no_grad():
                phi_next = self.p.drbfn.sample_phi(h_next)
                phi_next = phi_next.clamp(-self.p.phi_clamp, self.p.phi_clamp)

            # r = R/N + γΦ' - Φ  (with masking at terminal)
            r = (R_flat / N).unsqueeze(-1) + self.gamma * phi_next * mask_next_flat - phi

            # align = r · g
            align = (r * g_i).sum(dim=-1)  # (T_env,)

            # log p_BFN(Φ | s, a)
            log_p = self.p.drbfn.log_prob_phi(phi, h)  # (T_env,)

            aligns_list.append(align)
            log_ps_list.append(log_p)

        # Stack: (K, T_env)
        aligns = torch.stack(aligns_list)
        log_ps = torch.stack(log_ps_list)

        # K-sample normalize
        align_mean = aligns.mean(dim=0, keepdim=True)
        align_std = aligns.std(dim=0, keepdim=True) + 1e-8
        aligns_norm = (aligns - align_mean) / align_std

        # qadv weight
        weights = F.relu(aligns_norm)  # (K, T_env)

        # Skip if all zero (mean over batch)
        weight_sum = weights.sum()
        if weight_sum.item() < 1e-6:
            return 0.0, 0.0, 0.0  # skip this batch

        # Normalize weights PER (s,a) — sum over K dimension = 1 for each (s,a)
        weight_per_sa = weights.sum(dim=0, keepdim=True) + 1e-8  # (1, T_env)
        weights = weights / weight_per_sa

        # Loss = -Σ_k weight · log_p
        loss = -(weights * log_ps).sum(dim=0).mean()

        # σ stat (from g_net's last call to log_prob)
        # Just track align as proxy
        sigma_proxy = aligns.std().item()

        # Diagnostics: track Phi scale, gradient norm, log_p components
        with torch.no_grad():
            # Φ scale (from last sample)
            phi_scale = phi.abs().mean().item()
            # g_i scale
            g_i_scale = g_i.abs().mean().item()
            # Raw align (before normalize)
            raw_align_mean = aligns.mean().item()
            raw_align_std = aligns.std().item()
            # Log_p components
            log_p_mean = log_ps.mean().item()

        self.p.drbfn_optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.p.drbfn.parameters(), self.max_grad_norm).item()
        self.p.drbfn_optimizer.step()

        # Stash diagnostics on trainer for external access
        self._bfn_diag = {
            'phi_scale': phi_scale,
            'g_i_scale': g_i_scale,
            'raw_align_mean': raw_align_mean,
            'raw_align_std': raw_align_std,
            'log_p_mean': log_p_mean,
            'grad_norm': grad_norm,
        }

        return loss.item(), aligns_norm.mean().item(), sigma_proxy

    # ====================================================================
    # Train both Q_tot and BFN (after warmup)
    # ====================================================================
    def _train_components(self, buffer):
        qtot_loss, g_n_mean = self._train_qtot(buffer)
        drbfn_loss, align_mean, sigma = self._train_bfn(buffer)

        if self.train_step_count % 10 == 0:
            diag = getattr(self, '_bfn_diag', {})
            print(f"[v_qvpo step={self.train_step_count}] "
                  f"qtot_loss={qtot_loss:.4f} | "
                  f"drbfn_loss={drbfn_loss:.4f} | "
                  f"align_mean={align_mean:.4f} | "
                  f"g_n_mean={g_n_mean:.4f} | "
                  f"phi_scale={diag.get('phi_scale', 0):.4f} | "
                  f"g_i_scale={diag.get('g_i_scale', 0):.4f} | "
                  f"raw_align_mean={diag.get('raw_align_mean', 0):.4f} | "
                  f"raw_align_std={diag.get('raw_align_std', 0):.4f} | "
                  f"log_p_mean={diag.get('log_p_mean', 0):.4f} | "
                  f"grad_norm={diag.get('grad_norm', 0):.4f}",
                  flush=True)

        return {
            'qtot_loss': qtot_loss,
            'drbfn_loss': drbfn_loss,
            'align_mean': align_mean,
            'n_step_return_mean': g_n_mean,
        }
