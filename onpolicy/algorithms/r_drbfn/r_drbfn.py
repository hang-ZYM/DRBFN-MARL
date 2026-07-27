"""R_DRBFN Trainer: R-MAPPO + DRBFN credit assignment (full version with dual gate).

Data flow:
  train(buffer):
    1. Save original rewards
    2. _compute_gated_rewards(buffer) → gated_rewards (per-agent)
    3. Replace buffer.rewards with gated_rewards
    4. Recompute buffer.returns (GAE) with gated rewards
    5. super().train(buffer)  → standard PPO
    6. Restore original rewards and returns
    7. _train_drbfn_components(buffer) → train BFN, Q_i, Q_tot
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check


class R_DRBFN(R_MAPPO):
    """R-MAPPO + DRBFN credit assignment."""

    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.train_step_count = 0
        # For stats
        self._drbfn_stats = {
            'drbfn_loss': 0.0, 'qi_loss': 0.0, 'qtot_loss': 0.0,
            'delta_q': 0.0, 'confidence': 0.0, 'drbfn_active_ratio': 0.0,
        }

    @property
    def p(self):
        """Shortcut to policy (R_DRBFNPolicy)."""
        return self.policy

    def train(self, buffer, update_actor=True):
        """Extended train with DRBFN reward decomposition."""
        # Total env steps so far (for warmup check)
        total_steps = self.train_step_count * buffer.episode_length * buffer.n_rollout_threads
        in_warmup = total_steps < self.p.warmup_t

        # Save originals
        original_rewards = buffer.rewards.copy()
        original_returns = buffer.returns.copy()

        if not in_warmup:
            # Compute gated rewards
            gated_rewards, stats = self._compute_gated_rewards(buffer)
            buffer.rewards[:] = gated_rewards
            self._drbfn_stats.update(stats)
        # else: warmup, use original R/n (buffer.rewards already is R repeated)

        # Recompute returns with (gated) rewards
        # value_normalizer: use trainer's (inherited from R_MAPPO)
        value_normalizer = self.value_normalizer
        # Need next_value: buffer.value_preds[-1] holds it (set by prior compute_returns)
        next_value = buffer.value_preds[-1].copy()
        buffer.compute_returns(next_value, value_normalizer)

        # Standard PPO train
        train_info = super().train(buffer, update_actor)

        # Restore original rewards and returns (for next episode's data collection)
        buffer.rewards[:] = original_rewards
        buffer.returns[:] = original_returns
        # Recompute returns with original rewards (so buffer is clean)
        buffer.compute_returns(next_value, value_normalizer)

        # Train DRBFN components
        if not in_warmup:
            drbfn_info = self._train_drbfn_components(buffer)
            train_info.update(drbfn_info)
        else:
            # During warmup, still train Q_i and Q_tot with R/n
            self._train_warmup_q(buffer)

        self.train_step_count += 1
        return train_info

    # ====================================================================
    # Core: compute gated per-agent rewards
    # ====================================================================
    def _compute_gated_rewards(self, buffer):
        """Compute gated individual rewards from buffer.

        Returns:
            gated_rewards: np.ndarray (T, env, N, 1) per-agent gated reward
            stats: dict with delta_q, confidence, drbfn_active_ratio
        """
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length  # number of transition steps

        # --- Extract data from buffer (only first T steps) ---
        # share_obs: (T+1, env, N, dim) → take agent 0's share_obs (shared)
        share_obs_np = buffer.share_obs[:T, :, 0, :]  # (T, env, dim)
        obs_np = buffer.obs[:T]  # (T, env, N, obs_dim)
        actions_np = buffer.actions[:T]  # (T, env, N, 1)
        rewards_np = buffer.rewards[:T, :, 0, :]  # (T, env, 1) — take agent 0 (shared R)
        masks_np = buffer.masks[:T]  # (T, env, N, 1)
        terminated_np = 1.0 - buffer.masks[1:T+1, :, 0, :]  # (T, env, 1)

        T_, env, _ = share_obs_np.shape  # (T, env, dim)
        num_actions = self.p.num_actions

        # --- Convert to torch ---
        share_obs = check(share_obs_np).to(**tpdv)  # (T, env, dim)
        obs = check(obs_np).to(**tpdv)  # (T, env, N, obs_dim)
        actions = check(actions_np).to(**tpdv).long()  # (T, env, N, 1)
        rewards_R = check(rewards_np).to(**tpdv)  # (T, env, 1)
        terminated = check(terminated_np).to(**tpdv)  # (T, env, 1)

        share_obs_dim = share_obs.shape[-1]
        obs_dim = obs.shape[-1]

        # Flatten for BFN: (T*env, ...)
        T_env = T * env
        so_flat = share_obs.reshape(T_env, share_obs_dim)  # (T*env, dim)
        R_flat = rewards_R.reshape(T_env, 1)  # (T*env, 1)

        # Actions → one-hot → joint action flat
        actions_sq = actions.squeeze(-1)  # (T, env, N)
        actions_oh = F.one_hot(actions_sq, num_actions).float()  # (T, env, N, n_act)
        joint_a = actions_oh.reshape(T, env, N * num_actions)  # (T, env, N*n_act)
        joint_a_flat = joint_a.reshape(T_env, N * num_actions)

        # BFN input
        drbfn_input = torch.cat([so_flat, joint_a_flat], dim=-1)  # (T*env, raw_dim)

        # --- BFN inference (deterministic for policy, stochastic for Q_i) ---
        with torch.no_grad():
            r_result, w_var = self.p.drbfn.predict(drbfn_input, R_flat.squeeze(-1))
            # r_result: tuple of N tensors, each (T*env,)
            # w_var: (T*env, N-1)
            r_deterministic = torch.stack([r_result[i] for i in range(N)], dim=-1)  # (T*env, N)

            r_stoch_result = self.p.drbfn.sample(drbfn_input, R_flat.squeeze(-1))
            r_stochastic = torch.stack([r_stoch_result[i] for i in range(N)], dim=-1)  # (T*env, N)

            # Confidence from BFN variance
            w_var_mean = w_var.mean(dim=-1, keepdim=True)  # (T*env, 1)
            confidence = 1.0 / (1.0 + w_var_mean)  # (T*env, 1)

        # --- Q_i forward (compute A_DRBFN) ---
        agent_id = self.p.agent_id_eye.unsqueeze(0).unsqueeze(0).expand(T, env, N, N)  # (T, env, N, N)
        with torch.no_grad():
            qi_values = self.p.qi_net(share_obs, obs, agent_id)  # (T, env, N, n_act)
            a_drbfn = qi_values.argmax(dim=-1)  # (T, env, N)
            a_drbfn_oh = F.one_hot(a_drbfn, num_actions).float()  # (T, env, N, n_act)
            joint_drbfn = a_drbfn_oh.reshape(T, env, N * num_actions)

        # --- A_target from current policy actor (deterministic argmax) ---
        # Step-by-step forward with proper RNN state propagation from buffer
        with torch.no_grad():
            recurrent_N = self.policy.actor.rnn._recurrent_N if hasattr(self.policy.actor, 'rnn') else 1
            hidden_size = self.p.hidden_size

            # Get initial RNN states from buffer: (env, N, recurrent_N, hidden)
            rnn_states_actor = check(buffer.rnn_states[0]).to(**tpdv)

            # available_actions from buffer (SMAC specific)
            avail_all = None
            if buffer.available_actions is not None:
                avail_all = check(buffer.available_actions[:T]).to(**tpdv)  # (T, env, N, num_actions)

            # Masks from buffer for RNN reset
            masks_all = check(buffer.masks[:T]).to(**tpdv)  # (T, env, N, 1)

            # Step-by-step forward (propagates RNN hidden states correctly)
            a_target_list = []
            for t in range(T):
                obs_t = obs[t]  # (env, N, obs_dim)
                masks_t = masks_all[t]  # (env, N, 1)
                avail_t = avail_all[t] if avail_all is not None else None

                # Flatten to (env*N, ...) for actor forward
                obs_t_flat = obs_t.reshape(env * N, obs_dim)
                rnn_t_flat = rnn_states_actor.reshape(env * N, recurrent_N, hidden_size)
                masks_t_flat = masks_t.reshape(env * N, 1)
                avail_t_flat = avail_t.reshape(env * N, num_actions) if avail_t is not None else None

                # Actor deterministic forward
                a_t_flat, _, rnn_states_actor_new = self.policy.actor(
                    obs_t_flat, rnn_t_flat, masks_t_flat,
                    available_actions=avail_t_flat, deterministic=True
                )
                # a_t_flat: (env*N, 1)
                a_target_list.append(a_t_flat.reshape(env, N))
                # Update RNN states for next step
                rnn_states_actor = rnn_states_actor_new.reshape(env, N, recurrent_N, hidden_size)

            a_target = torch.stack(a_target_list, dim=0)  # (T, env, N)
            a_target_oh = F.one_hot(a_target.long(), num_actions).float()
            joint_target = a_target_oh.reshape(T, env, N * num_actions)

        # --- δQ via Q_tot target ---
        with torch.no_grad():
            # Q_tot input: share_obs (agent 0) + joint action
            q_drbfn = self.p.qtot_target(share_obs, joint_drbfn)  # (T, env, 1)
            q_target_val = self.p.qtot_target(share_obs, joint_target)  # (T, env, 1)
            delta_q = q_drbfn - q_target_val  # (T, env, 1)

        # --- Dual gate ---
        confidence_reshaped = confidence.reshape(T, env, 1)  # (T*env, 1) → (T, env, 1)
        gate = (delta_q > 0).float() * confidence_reshaped  # (T, env, 1)
        # BUG FIX: use full R (not R/n) for gate=0 fallback.
        # R/n shrinks reward signal by N×, crippling PPO for large N (e.g. MMM2 N=10).
        # MAPPO uses full R per agent, so we match that.
        uniform = R_flat  # (T*env, 1) → reshape (T, env, 1)
        uniform = uniform.reshape(T, env, 1)

        r_det = r_deterministic.reshape(T, env, N)  # (T, env, N)
        gated = gate * r_det + (1 - gate) * (uniform.expand(-1, -1, N))
        # gated: (T, env, N)

        # Expand to (T, env, N, 1) for buffer compatibility
        gated_rewards = gated.unsqueeze(-1).cpu().numpy()  # (T, env, N, 1)

        # Stats
        stats = {
            'delta_q': delta_q.mean().item(),
            'confidence': confidence.mean().item(),
            'drbfn_active_ratio': gate.mean().item(),
        }

        return gated_rewards, stats

    # ====================================================================
    # Train BFN + Q_i + Q_tot
    # ====================================================================
    def _train_drbfn_components(self, buffer):
        """Train BFN, Q_i, Q_tot networks."""
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        # Extract buffer data
        share_obs_np = buffer.share_obs[:T, :, 0, :]
        obs_np = buffer.obs[:T]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        masks_np = buffer.masks[:T]
        next_masks_np = buffer.masks[1:T+1]
        share_obs_next_np = buffer.share_obs[1:T+1, :, 0, :]
        obs_next_np = buffer.obs[1:T+1]

        T_, env, _ = share_obs_np.shape

        share_obs = check(share_obs_np).to(**tpdv)
        obs = check(obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)  # (T, env, 1)
        masks = check(masks_np).to(**tpdv)
        next_masks = check(next_masks_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        obs_next = check(obs_next_np).to(**tpdv)
        terminated = 1.0 - next_masks[:, :, 0, :]  # (T, env, 1)

        share_obs_dim = share_obs.shape[-1]
        obs_dim = obs.shape[-1]
        T_env = T * env

        # ---------- BFN training (every drbfn_update_interval steps) ----------
        drbfn_loss_val = 0.0
        if self.train_step_count % self.p.drbfn_update_interval == 0:
            # First: re-compute r_deterministic with current BFN
            actions_sq = actions.squeeze(-1)
            actions_oh = F.one_hot(actions_sq, num_actions).float()
            joint_a = actions_oh.reshape(T, env, N * num_actions)
            so_flat = share_obs.reshape(T_env, -1)
            joint_a_flat = joint_a.reshape(T_env, -1)
            R_flat = R.reshape(T_env)
            drbfn_input = torch.cat([so_flat, joint_a_flat], dim=-1)

            with torch.no_grad():
                r_result, _ = self.p.drbfn.predict(drbfn_input, R_flat)
                r_deterministic = torch.stack([r_result[i] for i in range(N)], dim=-1)  # (T*env, N)
                r_deterministic = r_deterministic.reshape(T, env, N)

                # Critic V (use policy.critic)
                # share_obs is (T, env, 81); pass directly, _critic_forward adds dummy agent dim
                v_all = self._critic_forward(share_obs, buffer.rnn_states_critic[:T, :, 0], masks[:, :, 0])
                v_next = self._critic_forward(share_obs_next, buffer.rnn_states_critic[1:T+1, :, 0], next_masks[:, :, 0])
                # v_all, v_next: (T, env)
                v_all_exp = v_all.unsqueeze(-1)  # (T, env, 1)
                v_next_exp = v_next.unsqueeze(-1)  # (T, env, 1)
                # Broadcast to N agents
                term_exp = terminated  # (T, env, 1)
                td_residuals = r_deterministic + self.gamma * v_next_exp * (1 - term_exp) - v_all_exp  # (T, env, N)
                w_target_all = F.softmax(td_residuals, dim=-1)  # (T, env, N)

            # BFN training: target w = first N-1 components mapped to [-1,1]
            D = self.p.drbfn.D
            h_encoded = self.p.drbfn.encode(drbfn_input).reshape(T, env, -1)
            w_for_bfn = (w_target_all[:, :, :D] * 2.0 - 1.0).clamp(-1, 1)

            loss_d, loss_e = self.p.drbfn.train_bfn(
                h_encoded.reshape(T_env, -1),
                w_for_bfn.reshape(T_env, -1)
            )
            drbfn_loss = loss_d + self.p.drbfn_beta * loss_e
            self.p.drbfn_optimizer.zero_grad()
            drbfn_loss.backward()
            nn.utils.clip_grad_norm_(self.p.drbfn.parameters(), self.max_grad_norm)
            self.p.drbfn_optimizer.step()
            drbfn_loss_val = drbfn_loss.item()

        # ---------- Q_i training ----------
        # Compute stochastic r_i for Q_i target
        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, env, N * num_actions)
        so_flat = share_obs.reshape(T_env, -1)
        joint_a_flat = joint_a.reshape(T_env, -1)
        R_flat = R.reshape(T_env)
        drbfn_input = torch.cat([so_flat, joint_a_flat], dim=-1)
        with torch.no_grad():
            r_stoch_result = self.p.drbfn.sample(drbfn_input, R_flat)
            r_stochastic = torch.stack([r_stoch_result[i] for i in range(N)], dim=-1).reshape(T, env, N)

            agent_id = self.p.agent_id_eye.unsqueeze(0).unsqueeze(0).expand(T, env, N, N)
            qi_next = self.p.qi_target(share_obs_next, obs_next, agent_id)  # (T, env, N, n_act)
            qi_next_max = qi_next.max(dim=-1)[0]  # (T, env, N)
            term_expand = terminated.expand(-1, -1, N)
            qi_td_target = r_stochastic + self.gamma * qi_next_max * (1 - term_expand)

        qi_values = self.p.qi_net(share_obs, obs, agent_id)
        qi_taken = qi_values.gather(-1, actions).squeeze(-1)  # (T, env, N)

        qi_loss = ((qi_taken - qi_td_target.detach()) ** 2).mean()
        self.p.qi_optimizer.zero_grad()
        qi_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qi_net.parameters(), self.max_grad_norm)
        self.p.qi_optimizer.step()
        self.p.soft_update_qi()
        qi_loss_val = qi_loss.item()

        # ---------- Q_tot training ----------
        with torch.no_grad():
            qi_next_greedy = qi_next.argmax(dim=-1)  # (T, env, N)
            qi_next_greedy_oh = F.one_hot(qi_next_greedy, num_actions).float()
            joint_next = qi_next_greedy_oh.reshape(T, env, N * num_actions)
            q_tot_next = self.p.qtot_target(share_obs_next, joint_next)  # (T, env, 1)
            qtot_td_target = R + self.gamma * q_tot_next * (1 - terminated)

        joint_taken = joint_a  # already computed
        q_tot_pred = self.p.qtot_net(share_obs, joint_taken)  # (T, env, 1)

        qtot_loss = ((q_tot_pred - qtot_td_target.detach()) ** 2).mean()
        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qtot_net.parameters(), self.max_grad_norm)
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()
        qtot_loss_val = qtot_loss.item()

        return {
            'drbfn_loss': drbfn_loss_val,
            'qi_loss': qi_loss_val,
            'qtot_loss': qtot_loss_val,
        }

    def _train_warmup_q(self, buffer):
        """During warmup, train Q_i and Q_tot with R/n (uniform)."""
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

        share_obs_np = buffer.share_obs[:T, :, 0, :]
        obs_np = buffer.obs[:T]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        next_masks_np = buffer.masks[1:T+1]
        share_obs_next_np = buffer.share_obs[1:T+1, :, 0, :]
        obs_next_np = buffer.obs[1:T+1]

        share_obs = check(share_obs_np).to(**tpdv)
        obs = check(obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        R = check(rewards_np).to(**tpdv)
        next_masks = check(next_masks_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        obs_next = check(obs_next_np).to(**tpdv)
        terminated = 1.0 - next_masks[:, :, 0, :]

        T_env = T * share_obs.shape[1]
        uniform_r = (R / N).expand(-1, -1, N)  # (T, env, N)

        # Q_i training with R/n
        env = share_obs.shape[1]
        agent_id = self.p.agent_id_eye.unsqueeze(0).unsqueeze(0).expand(T, env, N, N)
        with torch.no_grad():
            qi_next = self.p.qi_target(share_obs_next, obs_next, agent_id)
            qi_next_max = qi_next.max(dim=-1)[0]
            term_expand = terminated.expand(-1, -1, N)
            qi_td_target = uniform_r + self.gamma * qi_next_max * (1 - term_expand)

        qi_values = self.p.qi_net(share_obs, obs, agent_id)
        qi_taken = qi_values.gather(-1, actions).squeeze(-1)

        qi_loss = ((qi_taken - qi_td_target.detach()) ** 2).mean()
        self.p.qi_optimizer.zero_grad()
        qi_loss.backward()
        self.p.qi_optimizer.step()
        self.p.soft_update_qi()

        # Q_tot training with R
        with torch.no_grad():
            qi_next_greedy = qi_next.argmax(dim=-1)
            qi_next_greedy_oh = F.one_hot(qi_next_greedy, num_actions).float()
            joint_next = qi_next_greedy_oh.reshape(T, env, N * num_actions)
            q_tot_next = self.p.qtot_target(share_obs_next, joint_next)
            qtot_td_target = R + self.gamma * q_tot_next * (1 - terminated)

        actions_oh = F.one_hot(actions.squeeze(-1), num_actions).float()
        joint_taken = actions_oh.reshape(T, env, N * num_actions)
        q_tot_pred = self.p.qtot_net(share_obs, joint_taken)

        qtot_loss = ((q_tot_pred - qtot_td_target.detach()) ** 2).mean()
        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()

    def _critic_forward(self, share_obs, rnn_states, masks):
        """Forward critic to get V values. share_obs: (T, env, dim).

        Returns: (T, env) tensor of V values.
        """
        tpdv = dict(dtype=torch.float32, device=self.device)
        import numpy as np
        if isinstance(share_obs, np.ndarray):
            share_obs = check(share_obs).to(**tpdv)
        if isinstance(rnn_states, np.ndarray):
            rnn_states = check(rnn_states).to(**tpdv)
        if isinstance(masks, np.ndarray):
            masks = check(masks).to(**tpdv)

        # Save original shape
        T, env = share_obs.shape[0], share_obs.shape[1]

        # Flatten to batch: (T*env, ...)
        so_flat = share_obs.reshape(T * env, -1)  # (T*env, dim)
        rs_flat = rnn_states.reshape(T * env, *rnn_states.shape[2:])  # (T*env, recurrent_N, hidden)
        m_flat = masks.reshape(T * env, -1)  # (T*env, 1)

        # Critic forward
        values, _ = self.policy.critic(so_flat, rs_flat, m_flat)
        # values: (T*env, 1) → reshape back
        return values.reshape(T, env)
