"""R_DRBFN-v2 Trainer: generative per-agent reward via BFN.

Paradigm shift from v1:
  v1: decomposition (Σr_i = R hard constraint, softmax(TD) target, Q_i + dual gate)
  v2: generation    (Σr_i ≈ R soft prior, counterfactual target, multi-sample)

Data flow:
  train(buffer):
    1. Save original rewards
    2. _compute_generated_rewards(buffer) -> per-agent rewards via BFN
       - Sample K times from BFN, take mean
       - Add exploration bonus from sample variance
    3. Replace buffer.rewards with generated rewards
    4. Recompute buffer.returns (GAE)
    5. super().train(buffer) -> standard PPO
    6. Restore original rewards
    7. _train_components(buffer):
       - Train BFN with counterfactual target r_i^CF = Q_tot(s,a) - Q_tot(s, a_{-i}, a_i')
       - Train Q_tot with standard TD
       - Soft update Q_tot target

Conservation prior: KL implemented as ||Σr_i - R||^2 added to BFN loss (soft).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check


class R_DRBFN_v2(R_MAPPO):
    """R-MAPPO + DRBFN-v2 generative per-agent reward."""

    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.train_step_count = 0

        # v2 hyperparams (with defaults if not in args)
        self.K_samples = getattr(args, 'drbfn_k_samples', 4)
        self.lambda_conservation = getattr(args, 'drbfn_lambda_cons', 0.1)
        self.lambda_exploration = getattr(args, 'drbfn_lambda_exp', 0.01)
        self.lambda_shaping = getattr(args, 'drbfn_lambda_shaping', 1.0)
        self.drbfn_beta = args.drbfn_beta
        self.max_grad_norm = args.max_grad_norm

        # For stats
        self._drbfn_stats = {
            'drbfn_loss': 0.0,
            'qtot_loss': 0.0,
            'counterfactual_mean': 0.0,
            'r_variance': 0.0,
            'conservation_violation': 0.0,
        }

    @property
    def p(self):
        return self.policy

    def train(self, buffer, update_actor=True):
        """v2 with advantage shaping (not reward replacement).

        Key change from previous v2:
          - buffer.rewards stays as R (true environment reward)
          - GAE returns computed with R (PPO math preserved)
          - BFN-generated per-agent signal added as SHAPING to buffer.returns
          - Critic still trains on shaped returns (approximation, but actor
            advantage is now correctly structured)
        """
        total_steps = self.train_step_count * buffer.episode_length * buffer.n_rollout_threads
        in_warmup = total_steps < self.p.warmup_t

        # Standard GAE with original R
        value_normalizer = self.value_normalizer
        next_value = buffer.value_preds[-1].copy()
        buffer.compute_returns(next_value, value_normalizer)

        # Save original returns (we will add shaping)
        original_returns = buffer.returns.copy()

        if not in_warmup:
            # Compute BFN-generated per-agent shaping signal
            shaping_signal, stats = self._compute_shaping_signal(buffer)
            # shaping_signal: (T, env, N) torch tensor

            # Add shaping to buffer.returns[:-1] (broadcast to agent dim)
            # returns shape: (T+1, env, N, 1) numpy
            alpha = self.lambda_shaping
            shaping_np = shaping_signal.cpu().numpy()  # (T, env, N)
            shaping_centered = shaping_np - shaping_np.mean(axis=2, keepdims=True)
            # Broadcast (T, env, N, 1) by adding last dim
            buffer.returns[:-1] = buffer.returns[:-1] + alpha * shaping_centered[..., np.newaxis]
            self._drbfn_stats.update(stats)

        # Standard PPO train (uses shaped returns internally)
        train_info = super().train(buffer, update_actor)

        # Restore original returns
        buffer.returns[:] = original_returns

        # Train BFN + Q_tot
        if not in_warmup:
            comp_info = self._train_components(buffer)
            train_info.update(comp_info)
        else:
            self._train_warmup_qtot(buffer)

        self.train_step_count += 1
        return train_info

    # ====================================================================
    # Core 1 (new): compute per-agent shaping signal (BFN counterfactual_r)
    # ====================================================================
    def _compute_shaping_signal(self, buffer):
        """Sample from BFN to get per-agent counterfactual signal for shaping.

        Returns:
            shaping_signal: torch.Tensor (T, env, N) on device
            stats: dict
        """
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length

        share_obs_np = buffer.share_obs[:T, :, 0, :]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]

        T_, env, _ = share_obs_np.shape
        num_actions = self.p.num_actions

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)

        T_env = T * env

        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, env, N * num_actions)

        so_flat = share_obs.reshape(T_env, -1)
        joint_a_flat = joint_a.reshape(T_env, -1)
        R_flat = R.reshape(T_env, 1)
        drbfn_input = torch.cat([so_flat, joint_a_flat, R_flat], dim=-1)

        # Multi-sample from BFN, take mean (use as deterministic shaping)
        with torch.no_grad():
            r_samples = self.p.drbfn.sample_k(drbfn_input, K=self.K_samples)
            r_mean = r_samples.mean(dim=1)  # (T*env, N)
            r_var = r_samples.var(dim=1)

        r_mean = r_mean.reshape(T, env, N)
        r_var = r_var.reshape(T, env, N)

        conservation_violation = (r_mean.sum(dim=-1) - R.squeeze(-1)).pow(2).mean().item()

        stats = {
            'r_variance': r_var.mean().item(),
            'conservation_violation': conservation_violation,
            'shaping_abs_mean': r_mean.abs().mean().item(),
        }

        if self.train_step_count % 10 == 0:
            print(f"[v2-shape step={self.train_step_count}] "
                  f"R_mean={R.mean().item():.4f} | "
                  f"r_mean(mean,min,max)=({r_mean.mean().item():.4f},{r_mean.min().item():.4f},{r_mean.max().item():.4f}) | "
                  f"r_var_mean={r_var.mean().item():.6f} | "
                  f"shaping_abs={r_mean.abs().mean().item():.4f} | "
                  f"alpha={self.lambda_shaping}",
                  flush=True)

        return r_mean, stats

    # ====================================================================
    # Core 1: compute generated per-agent rewards (multi-sample + bonus)
    # ====================================================================
    def _compute_generated_rewards(self, buffer):
        """Sample K times from BFN, take mean + exploration bonus.

        Returns:
            generated_rewards: np.ndarray (T, env, N, 1)
            stats: dict
        """
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length

        # Extract buffer data
        share_obs_np = buffer.share_obs[:T, :, 0, :]      # (T, env, dim)
        actions_np = buffer.actions[:T]                    # (T, env, N, 1)
        rewards_np = buffer.rewards[:T, :, 0, :]           # (T, env, 1)

        T_, env, _ = share_obs_np.shape
        num_actions = self.p.num_actions

        share_obs = check(share_obs_np).to(**tpdv)         # (T, env, dim)
        actions = check(actions_np).to(**tpdv).long()      # (T, env, N, 1)
        R = check(rewards_np).to(**tpdv)                   # (T, env, 1)

        T_env = T * env

        # Build joint action onehot
        actions_sq = actions.squeeze(-1)                   # (T, env, N)
        actions_oh = F.one_hot(actions_sq, num_actions).float()  # (T, env, N, n_act)
        joint_a = actions_oh.reshape(T, env, N * num_actions)    # (T, env, N*n_act)

        # Flatten for BFN
        so_flat = share_obs.reshape(T_env, -1)             # (T*env, dim)
        joint_a_flat = joint_a.reshape(T_env, -1)          # (T*env, N*n_act)
        R_flat = R.reshape(T_env, 1)                       # (T*env, 1)

        # v2 conditioning: [share_obs, joint_a, R]
        drbfn_input = torch.cat([so_flat, joint_a_flat, R_flat], dim=-1)

        # Multi-sample from BFN
        with torch.no_grad():
            r_samples = self.p.drbfn.sample_k(drbfn_input, K=self.K_samples)
            # r_samples: (T*env, K, N)

            r_mean = r_samples.mean(dim=1)                 # (T*env, N)
            r_var = r_samples.var(dim=1)                   # (T*env, N)

            # Exploration bonus: λ_exp · Var_k(r_i^k)
            exploration_bonus = self.lambda_exploration * r_var  # (T*env, N)

            r_final = r_mean + exploration_bonus           # (T*env, N)

        # Reshape back
        r_final = r_final.reshape(T, env, N)               # (T, env, N)

        # Expand to (T, env, N, 1) for buffer
        generated_rewards = r_final.unsqueeze(-1).cpu().numpy()

        # Stats
        conservation_violation = (r_mean.sum(dim=-1) - R_flat.squeeze(-1)).pow(2).mean().item()

        stats = {
            'r_variance': r_var.mean().item(),
            'conservation_violation': conservation_violation,
            'exploration_bonus_mean': exploration_bonus.mean().item(),
        }

        # Debug: print internals every 10 steps
        if self.train_step_count % 10 == 0:
            print(f"[v2-debug step={self.train_step_count}] "
                  f"R_mean={R.mean().item():.4f}, R_abs_mean={R.abs().mean().item():.4f} | "
                  f"r_mean(mean,min,max)=({r_mean.mean().item():.4f},{r_mean.min().item():.4f},{r_mean.max().item():.4f}) | "
                  f"r_var_mean={r_var.mean().item():.6f} | "
                  f"sum_r_mean={r_mean.sum(dim=-1).mean().item():.4f} | "
                  f"gen_r_abs_mean={abs(generated_rewards).mean():.4f}",
                  flush=True)

        return generated_rewards, stats

    # ====================================================================
    # Core 2: train BFN with counterfactual target + Q_tot with TD
    # ====================================================================
    def _train_components(self, buffer):
        """Train BFN (counterfactual target) and Q_tot (TD)."""
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        # Extract data
        share_obs_np = buffer.share_obs[:T, :, 0, :]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        masks_np = buffer.masks[:T]
        next_masks_np = buffer.masks[1:T+1]
        share_obs_next_np = buffer.share_obs[1:T+1, :, 0, :]

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)
        next_masks = check(next_masks_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        terminated = 1.0 - next_masks[:, :, 0, :]          # (T, env, 1)

        T_, env, _ = share_obs_np.shape
        T_env = T * env

        # Joint action onehot
        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, env, N * num_actions)

        so_flat = share_obs.reshape(T_env, -1)
        joint_a_flat = joint_a.reshape(T_env, -1)
        R_flat = R.reshape(T_env, 1)
        drbfn_input = torch.cat([so_flat, joint_a_flat, R_flat], dim=-1)

        # ---------- Step 1: Compute counterfactual target ----------
        # r_i^target = softmax((Q_tot(s,a) - Q_tot(s, a_{-i}, a_i')) / τ) · R
        with torch.no_grad():
            counterfactual_r = self._compute_counterfactual_target(
                share_obs, actions, joint_a, R
            )
            # counterfactual_r: (T, env, N)
            counterfactual_r_flat = counterfactual_r.reshape(T_env, N)

        # ---------- Step 2: Train BFN (every drbfn_update_interval steps) ----------
        drbfn_loss_val = 0.0
        if self.train_step_count % self.p.drbfn_update_interval == 0:
            h_encoded = self.p.drbfn.encode(drbfn_input)

            loss_d, loss_e = self.p.drbfn.train_bfn(
                h_encoded, counterfactual_r_flat
            )

            # Conservation soft prior: ||Σr_i - R||^2
            # Use BFN's predicted r to compute the violation
            r_pred_mean, _ = self.p.drbfn.predict_r(h_encoded)
            sum_r = r_pred_mean.sum(dim=-1, keepdim=True)  # (T*env, 1)
            conservation_loss = (sum_r - R_flat).pow(2).mean()

            drbfn_loss = (loss_d
                          + self.drbfn_beta * loss_e
                          + self.lambda_conservation * conservation_loss)

            self.p.drbfn_optimizer.zero_grad()
            drbfn_loss.backward()
            nn.utils.clip_grad_norm_(self.p.drbfn.parameters(), self.max_grad_norm)
            self.p.drbfn_optimizer.step()
            drbfn_loss_val = drbfn_loss.item()

        # ---------- Step 3: Train Q_tot (standard TD, true SARSA) ----------
        # Use real a_{t+1} from buffer.actions[1:T] (drops last transition
        # since buffer.actions has length T, no a_T available).
        T_eff = T - 1
        share_obs_cur = share_obs[:T_eff]
        joint_a_cur = joint_a[:T_eff]
        R_cur = R[:T_eff]
        share_obs_next_eff = share_obs_next[:T_eff]
        term_eff = terminated[:T_eff]
        actions_next_np = buffer.actions[1:T]  # (T_eff, env, N, 1)
        actions_next = check(actions_next_np).to(**tpdv).long()
        actions_next_oh = F.one_hot(actions_next.squeeze(-1), num_actions).float()
        joint_a_next = actions_next_oh.reshape(T_eff, env, N * num_actions)

        with torch.no_grad():
            q_tot_next = self.p.qtot_target(share_obs_next_eff, joint_a_next)
            qtot_td_target = R_cur + self.gamma * q_tot_next * (1 - term_eff)

        q_tot_pred = self.p.qtot_net(share_obs_cur, joint_a_cur)
        qtot_loss = ((q_tot_pred - qtot_td_target.detach()) ** 2).mean()

        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qtot_net.parameters(), self.max_grad_norm)
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()
        qtot_loss_val = qtot_loss.item()

        # Debug: print Q_tot and counterfactual stats
        if self.train_step_count % 10 == 0:
            with torch.no_grad():
                q_factual_dbg = self.p.qtot_target(share_obs_cur[:5], joint_a_cur[:5])
                print(f"[v2-qdbg step={self.train_step_count}] "
                      f"Q_tot(factual) sample: mean={q_factual_dbg.mean().item():.4f}, "
                      f"min={q_factual_dbg.min().item():.4f}, max={q_factual_dbg.max().item():.4f} | "
                      f"counterfactual_r(cur step) mean={counterfactual_r.mean().item():.4f}, "
                      f"min={counterfactual_r.min().item():.4f}, max={counterfactual_r.max().item():.4f} | "
                      f"drbfn_loss={drbfn_loss_val:.4f}, qtot_loss={qtot_loss_val:.4f}",
                      flush=True)

        return {
            'drbfn_loss': drbfn_loss_val,
            'qtot_loss': qtot_loss_val,
            'counterfactual_mean': counterfactual_r.mean().item(),
        }

    # ====================================================================
    # Counterfactual target computation (batched, no loop over agents)
    # ====================================================================
    def _compute_counterfactual_target(self, share_obs, actions, joint_a, R):
        """Compute per-agent reward target via counterfactual weighting.

        v2 design (per DRBFN_v2_公式与流程.md §2.2):
            Δ_i^CF = Q_tot(s, a) - Q_tot(s, a_{-i}, a_i')    # counterfactual advantage
            w_i    = softmax(Δ_i^CF / τ)                     # weight in [0,1], Σw=1
            r_i^target = w_i · R                              # same scale as R, Σr=R

        Using Δ_i^CF directly as BFN target is WRONG (scale mismatch with R).
        Softmax-weighted redistribution keeps reward scale and counters the issue.

        Reference action a_i' is uniform random (COMA default baseline).
        Batched implementation: construct N modified joint actions, forward once.

        Args:
            share_obs: (T, env, dim)
            actions:   (T, env, N, 1) - taken actions (for shape ref)
            joint_a:   (T, env, N*n_act) - onehot of taken joint action
            R:         (T, env, 1) - global reward

        Returns:
            target_r: (T, env, N) - per-agent reward target (Σ over N ≈ R)
        """
        device = self.device
        N = self.num_agents
        num_actions = self.p.num_actions
        T, env = share_obs.shape[0], share_obs.shape[1]
        T_env = T * env
        tau = self.p.cf_temperature

        # Q_tot(s, a) - the factual value
        q_tot_factual = self.p.qtot_target(share_obs, joint_a)  # (T, env, 1)

        # Sample reference actions a_i' ~ uniform
        reference_actions = self._sample_reference_actions(share_obs, actions)
        # reference_actions: (T, env, N, n_act) onehot

        # Construct N counterfactual joint actions:
        # joint_a_cf[i] = joint_a but with agent i's action replaced by reference
        actions_oh = joint_a.reshape(T, env, N, num_actions)
        eye = torch.eye(N, device=device).view(N, 1, 1, N, 1)  # (N,1,1,N,1)
        actions_oh_exp = actions_oh.unsqueeze(0).expand(N, T, env, N, num_actions)
        reference_exp = reference_actions.unsqueeze(0).expand(N, T, env, N, num_actions)
        joint_a_cf = torch.where(eye.bool(), reference_exp, actions_oh_exp)
        # joint_a_cf: (N, T, env, N, n_act)

        # Batched forward: (N*T*env, N*n_act) -> (N*T*env, 1)
        joint_a_cf_flat = joint_a_cf.reshape(N * T_env, N * num_actions)
        share_obs_exp = share_obs.unsqueeze(0).expand(N, T, env, -1).reshape(N * T_env, -1)
        q_tot_cf_all = self.p.qtot_target(share_obs_exp, joint_a_cf_flat)
        q_tot_cf_all = q_tot_cf_all.reshape(N, T, env)  # (N, T, env)

        # Δ_i^CF = Q_tot(s,a) - Q_tot(s, a_{-i}, a_i')
        q_factual_broadcast = q_tot_factual.squeeze(-1).unsqueeze(0).expand(N, T, env)
        delta_cf = q_factual_broadcast - q_tot_cf_all  # (N, T, env)

        # Permute to (T, env, N) for softmax over agent dim
        delta_cf = delta_cf.permute(1, 2, 0).contiguous()  # (T, env, N)

        # Softmax weights (over agents), temperature-scaled
        weights = F.softmax(delta_cf / tau, dim=-1)  # (T, env, N), Σ_i w_i = 1

        # Reward target: w_i · R, same scale as R, strictly conserved
        R_broadcast = R.expand(-1, -1, N)  # (T, env, N)
        target_r = weights * R_broadcast    # (T, env, N)

        return target_r

    def _sample_reference_actions(self, share_obs, actions):
        """Sample reference actions a_i' for counterfactual computation.

        v2 simple version: uniform random over all actions.
        TODO: sample from actor's policy for policy-consistent baseline.

        Args:
            share_obs: (T, env, dim)
            actions:   (T, env, N, 1) - taken actions (unused for uniform)

        Returns:
            reference: (T, env, N, n_act) onehot
        """
        T, env = actions.shape[0], actions.shape[1]
        N = self.num_agents
        num_actions = self.p.num_actions
        device = actions.device

        # Uniform sample
        ref = torch.randint(0, num_actions, (T, env, N), device=device)
        ref_oh = F.one_hot(ref, num_actions).float()
        return ref_oh

    # ====================================================================
    # Warmup: just train Q_tot with R (so it's ready when BFN kicks in)
    # ====================================================================
    def _train_warmup_qtot(self, buffer):
        """During warmup, only train Q_tot with R. True SARSA (uses real a_{t+1})."""
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        # buffer.actions has shape (T, ...) without +1.
        # To get real next action a_{t+1}, we use t in [0, T-2]:
        #   share_obs[t], actions[t], rewards[t], share_obs_next[t+1], actions_next[t+1]
        # Drop the last transition (no a_T available). Effective length T-1.
        T_eff = T - 1

        share_obs_np = buffer.share_obs[:T_eff, :, 0, :]
        actions_np = buffer.actions[:T_eff]
        rewards_np = buffer.rewards[:T_eff, :, 0, :]
        share_obs_next_np = buffer.share_obs[1:T_eff+1, :, 0, :]
        actions_next_np = buffer.actions[1:T_eff+1]
        next_masks_np = buffer.masks[1:T_eff+1]

        share_obs = check(share_obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        actions_next = check(actions_next_np).to(**tpdv).long()
        next_masks = check(next_masks_np).to(**tpdv)
        terminated = 1.0 - next_masks[:, :, 0, :]

        env = share_obs_np.shape[1]

        actions_oh = F.one_hot(actions.squeeze(-1), num_actions).float()
        joint_a = actions_oh.reshape(T_eff, env, N * num_actions)
        actions_next_oh = F.one_hot(actions_next.squeeze(-1), num_actions).float()
        joint_a_next = actions_next_oh.reshape(T_eff, env, N * num_actions)

        with torch.no_grad():
            q_tot_next = self.p.qtot_target(share_obs_next, joint_a_next)
            qtot_td_target = R + self.gamma * q_tot_next * (1 - terminated)

        q_tot_pred = self.p.qtot_net(share_obs, joint_a)
        qtot_loss = ((q_tot_pred - qtot_td_target.detach()) ** 2).mean()

        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qtot_net.parameters(), self.max_grad_norm)
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()
