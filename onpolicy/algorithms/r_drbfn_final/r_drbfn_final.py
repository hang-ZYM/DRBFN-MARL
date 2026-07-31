"""R_DRBFN_FINAL Trainer: v1 + n-step Q_tot target.

The ONLY change from v1: Q_tot target uses n-step return instead of 1-step TD.
Everything else (BFN self-supervised target, stick-breaking, Dual Gate, Q_i)
is preserved exactly as v1.

Rationale:
- v1 is best on 5m_vs_6m/3m/2c_vs_64zg (60%/99%/88%) — keep its robustness
- v1 fails on MMM2 because Q_tot learns poorly → counterfactual δQ_i is wrong
- v3 shows n-step return makes Q_tot learn better on MMM2
- v_final = v1 + n-step Q_tot: keeps v1 robustness, gains v3's MMM2 capability
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO
from onpolicy.algorithms.utils.util import check


class R_DRBFN_FINAL(R_MAPPO):
    """v_final = v1 + n-step Q_tot target."""

    def __init__(self, args, policy, num_agents, device=torch.device("cpu")):
        super().__init__(args, policy, device)
        self.num_agents = num_agents
        self.gamma = args.gamma
        self.train_step_count = 0

        self.n_step = getattr(args, 'drbfn_n_step', 5)
        self.drbfn_beta = args.drbfn_beta
        self.max_grad_norm = args.max_grad_norm

        self._stats = {
            'drbfn_loss': 0.0, 'qi_loss': 0.0, 'qtot_loss': 0.0,
            'delta_q': 0.0, 'confidence': 0.0, 'drbfn_active_ratio': 0.0,
            'n_step_return_mean': 0.0,
        }

    @property
    def p(self):
        return self.policy

    def train(self, buffer, update_actor=True):
        total_steps = self.train_step_count * buffer.episode_length * buffer.n_rollout_threads
        in_warmup = total_steps < self.p.warmup_t

        original_rewards = buffer.rewards.copy()
        original_returns = buffer.returns.copy()

        if not in_warmup:
            gated_rewards, stats = self._compute_gated_rewards(buffer)
            buffer.rewards[:] = gated_rewards
            self._stats.update(stats)

        value_normalizer = self.value_normalizer
        next_value = buffer.value_preds[-1].copy()
        buffer.compute_returns(next_value, value_normalizer)

        train_info = super().train(buffer, update_actor)

        buffer.rewards[:] = original_rewards
        buffer.returns[:] = original_returns
        buffer.compute_returns(next_value, value_normalizer)

        if not in_warmup:
            comp_info = self._train_components(buffer)
            train_info.update(comp_info)
        else:
            self._train_warmup_q(buffer)

        self.train_step_count += 1
        return train_info

    # ====================================================================
    # Compute gated per-agent rewards (identical to v1)
    # ====================================================================
    def _compute_gated_rewards(self, buffer):
        """v1 dual gate: BFN r_i when Q_tot says δQ>0 and BFN confident, else R."""
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length

        share_obs_np = buffer.share_obs[:T, :, 0, :]
        obs_np = buffer.obs[:T]
        actions_np = buffer.actions[:T]
        rewards_np = buffer.rewards[:T, :, 0, :]
        masks_np = buffer.masks[:T]
        terminated_np = 1.0 - buffer.masks[1:T+1, :, 0, :]

        T_, env, _ = share_obs_np.shape
        num_actions = self.p.num_actions

        share_obs = check(share_obs_np).to(**tpdv)
        obs = check(obs_np).to(**tpdv)
        actions = check(actions_np).to(**tpdv).long()
        rewards_R = check(rewards_np).to(**tpdv)
        terminated = check(terminated_np).to(**tpdv)

        T_env = T * env
        so_flat = share_obs.reshape(T_env, -1)
        R_flat = rewards_R.reshape(T_env, 1)

        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, env, N * num_actions)
        joint_a_flat = joint_a.reshape(T_env, -1)

        drbfn_input = torch.cat([so_flat, joint_a_flat], dim=-1)

        with torch.no_grad():
            r_result, w_var = self.p.drbfn.predict(drbfn_input, R_flat.squeeze(-1))
            r_deterministic = torch.stack([r_result[i] for i in range(N)], dim=-1)
            r_stoch_result = self.p.drbfn.sample(drbfn_input, R_flat.squeeze(-1))
            r_stochastic = torch.stack([r_stoch_result[i] for i in range(N)], dim=-1)
            w_var_mean = w_var.mean(dim=-1, keepdim=True)
            confidence = 1.0 / (1.0 + w_var_mean)

        agent_id = self.p.agent_id_eye.unsqueeze(0).unsqueeze(0).expand(T, env, N, N)
        with torch.no_grad():
            qi_values = self.p.qi_net(share_obs, obs, agent_id)
            a_drbfn = qi_values.argmax(dim=-1)
            a_drbfn_oh = F.one_hot(a_drbfn, num_actions).float()
            joint_drbfn = a_drbfn_oh.reshape(T, env, N * num_actions)

        with torch.no_grad():
            recurrent_N = self.policy.actor.rnn._recurrent_N if hasattr(self.policy.actor, 'rnn') else 1
            hidden_size = self.p.hidden_size
            rnn_states_actor = check(buffer.rnn_states[0]).to(**tpdv)

            avail_all = None
            if buffer.available_actions is not None:
                avail_all = check(buffer.available_actions[:T]).to(**tpdv)
            masks_all = check(buffer.masks[:T]).to(**tpdv)

            a_target_list = []
            for t in range(T):
                obs_t = obs[t]
                masks_t = masks_all[t]
                avail_t = avail_all[t] if avail_all is not None else None

                obs_t_flat = obs_t.reshape(env * N, -1)
                rnn_t_flat = rnn_states_actor.reshape(env * N, recurrent_N, hidden_size)
                masks_t_flat = masks_t.reshape(env * N, 1)
                avail_t_flat = avail_t.reshape(env * N, num_actions) if avail_t is not None else None

                a_t_flat, _, rnn_states_actor_new = self.policy.actor(
                    obs_t_flat, rnn_t_flat, masks_t_flat,
                    available_actions=avail_t_flat, deterministic=True
                )
                a_target_list.append(a_t_flat.reshape(env, N))
                rnn_states_actor = rnn_states_actor_new.reshape(env, N, recurrent_N, hidden_size)

            a_target = torch.stack(a_target_list, dim=0)
            a_target_oh = F.one_hot(a_target.long(), num_actions).float()
            joint_target = a_target_oh.reshape(T, env, N * num_actions)

        with torch.no_grad():
            q_drbfn = self.p.qtot_target(share_obs, joint_drbfn)
            q_target_val = self.p.qtot_target(share_obs, joint_target)
            delta_q = q_drbfn - q_target_val

        confidence_reshaped = confidence.reshape(T, env, 1)
        gate = (delta_q > 0).float() * confidence_reshaped

        # R/n bug fixed: use full R for gate=0 fallback
        uniform = R_flat.reshape(T, env, 1)

        r_det = r_deterministic.reshape(T, env, N)
        gated = gate * r_det + (1 - gate) * (uniform.expand(-1, -1, N))
        gated_rewards = gated.unsqueeze(-1).cpu().numpy()

        stats = {
            'delta_q': delta_q.mean().item(),
            'confidence': confidence.mean().item(),
            'drbfn_active_ratio': gate.mean().item(),
        }

        return gated_rewards, stats

    # ====================================================================
    # *** v_final ONLY CHANGE: n-step return for Q_tot target ***
    # ====================================================================
    def _compute_n_step_return(self, buffer, share_obs, joint_a, R, terminated):
        """G_t^n = Σ γ^k R_{t+k} + γ^n Q_target(s_{t+n}, a_{t+n}).

        For t+n beyond episode end (terminated), no bootstrap.
        """
        device = self.device
        T, env, _ = share_obs.shape
        n = self.n_step
        gamma = self.gamma

        discounts = torch.tensor([gamma ** k for k in range(n)],
                                  dtype=torch.float32, device=device).view(1, 1, n)

        G_n = torch.zeros_like(R)
        rewards_full = check(buffer.rewards[:T, :, 0, :]).to(device=device, dtype=torch.float32)
        next_masks_full = check(buffer.masks[1:T+1, :, 0, :]).to(device=device, dtype=torch.float32)
        share_obs_full = check(buffer.share_obs[:T+1, :, 0, :]).to(device=device, dtype=torch.float32)

        # Sum of discounted rewards over next n steps
        for k in range(n):
            if k == 0:
                r_k = R
            else:
                r_k = torch.zeros_like(R)
                if k < T:
                    r_k[:T-k] = rewards_full[k:T]
            G_n = G_n + discounts[:, :, k] * r_k

        # Bootstrap term γ^n Q_target(s_{t+n}, a_{t+n})
        # Zero out if episode ended within window OR t+n >= T
        term_cum = torch.zeros_like(terminated)
        for k in range(n):
            if k == 0:
                term_cum = term_cum + terminated
            else:
                t_shifted = torch.zeros_like(terminated)
                if k < T:
                    t_shifted[:T-k] = terminated[k:T]
                term_cum = term_cum + t_shifted
        valid_bootstrap = (1.0 - term_cum.clamp(max=1.0))

        Q_next = torch.zeros_like(R)
        if n < T:
            so_next = share_obs_full[n:T]
            ja_next = joint_a[n:T]
            q_next_val = self.p.qtot_target(so_next, ja_next)
            Q_next[:T-n] = q_next_val

        bootstrap_factor = gamma ** n
        G_n = G_n + bootstrap_factor * Q_next * valid_bootstrap

        return G_n

    # ====================================================================
    # Train BFN + Q_i + Q_tot
    # ====================================================================
    def _train_components(self, buffer):
        device = self.device
        tpdv = dict(dtype=torch.float32, device=device)
        N = self.num_agents
        T = buffer.episode_length
        num_actions = self.p.num_actions

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
        R = check(rewards_np).to(**tpdv)
        masks = check(masks_np).to(**tpdv)
        next_masks = check(next_masks_np).to(**tpdv)
        share_obs_next = check(share_obs_next_np).to(**tpdv)
        obs_next = check(obs_next_np).to(**tpdv)
        terminated = 1.0 - next_masks[:, :, 0, :]

        T_env = T * env

        actions_sq = actions.squeeze(-1)
        actions_oh = F.one_hot(actions_sq, num_actions).float()
        joint_a = actions_oh.reshape(T, env, N * num_actions)
        so_flat = share_obs.reshape(T_env, -1)
        joint_a_flat = joint_a.reshape(T_env, -1)
        R_flat = R.reshape(T_env)

        drbfn_input = torch.cat([so_flat, joint_a_flat], dim=-1)

        # ---------- BFN training (v1 self-supervised target) ----------
        drbfn_loss_val = 0.0
        if self.train_step_count % self.p.drbfn_update_interval == 0:
            with torch.no_grad():
                r_result, _ = self.p.drbfn.predict(drbfn_input, R_flat)
                r_deterministic = torch.stack([r_result[i] for i in range(N)], dim=-1).reshape(T, env, N)

                v_all = self._critic_forward(share_obs, buffer.rnn_states_critic[:T, :, 0], masks[:, :, 0])
                v_next = self._critic_forward(share_obs_next, buffer.rnn_states_critic[1:T+1, :, 0], next_masks[:, :, 0])
                v_all_exp = v_all.unsqueeze(-1)
                v_next_exp = v_next.unsqueeze(-1)
                term_exp = terminated
                td_residuals = r_deterministic + self.gamma * v_next_exp * (1 - term_exp) - v_all_exp
                w_target_all = F.softmax(td_residuals, dim=-1)

            D = self.p.drbfn.D
            h_encoded = self.p.drbfn.encode(drbfn_input).reshape(T, env, -1)
            w_for_bfn = (w_target_all[:, :, :D] * 2.0 - 1.0).clamp(-1, 1)

            loss_d, loss_e = self.p.drbfn.train_bfn(
                h_encoded.reshape(T_env, -1),
                w_for_bfn.reshape(T_env, -1)
            )
            drbfn_loss = loss_d + self.drbfn_beta * loss_e
            self.p.drbfn_optimizer.zero_grad()
            drbfn_loss.backward()
            nn.utils.clip_grad_norm_(self.p.drbfn.parameters(), self.max_grad_norm)
            self.p.drbfn_optimizer.step()
            drbfn_loss_val = drbfn_loss.item()

        # ---------- Q_i training (same as v1) ----------
        with torch.no_grad():
            r_stoch_result = self.p.drbfn.sample(drbfn_input, R_flat)
            r_stochastic = torch.stack([r_stoch_result[i] for i in range(N)], dim=-1).reshape(T, env, N)

            agent_id = self.p.agent_id_eye.unsqueeze(0).unsqueeze(0).expand(T, env, N, N)
            qi_next = self.p.qi_target(share_obs_next, obs_next, agent_id)
            qi_next_max = qi_next.max(dim=-1)[0]
            term_expand = terminated.expand(-1, -1, N)
            qi_td_target = r_stochastic + self.gamma * qi_next_max * (1 - term_expand)

        qi_values = self.p.qi_net(share_obs, obs, agent_id)
        qi_taken = qi_values.gather(-1, actions).squeeze(-1)
        qi_loss = ((qi_taken - qi_td_target.detach()) ** 2).mean()
        self.p.qi_optimizer.zero_grad()
        qi_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qi_net.parameters(), self.max_grad_norm)
        self.p.qi_optimizer.step()
        self.p.soft_update_qi()

        # ---------- *** v_final: Q_tot training with n-step return *** ----------
        # Compute n-step return G_t^n
        G_n = self._compute_n_step_return(buffer, share_obs, joint_a, R, terminated)
        # G_n: (T, env, 1) - same scale as cumulative reward

        q_tot_pred = self.p.qtot_net(share_obs, joint_a)
        qtot_loss = ((q_tot_pred - G_n.detach()) ** 2).mean()

        self.p.qtot_optimizer.zero_grad()
        qtot_loss.backward()
        nn.utils.clip_grad_norm_(self.p.qtot_net.parameters(), self.max_grad_norm)
        self.p.qtot_optimizer.step()
        self.p.soft_update_qtot()

        if self.train_step_count % 10 == 0:
            print(f"[v_final step={self.train_step_count}] "
                  f"R_mean={R.mean().item():.4f} | "
                  f"G_n_mean={G_n.mean().item():.4f} (n={self.n_step}) | "
                  f"q_tot_pred={q_tot_pred.mean().item():.4f} | "
                  f"drbfn_loss={drbfn_loss_val:.4f}, qi_loss={qi_loss.item():.4f}, "
                  f"qtot_loss={qtot_loss.item():.4f}",
                  flush=True)

        return {
            'drbfn_loss': drbfn_loss_val,
            'qi_loss': qi_loss.item(),
            'qtot_loss': qtot_loss.item(),
            'n_step_return_mean': G_n.mean().item(),
        }

    def _train_warmup_q(self, buffer):
        """Warmup: train Q_i and Q_tot with R/n and R (v1 style, 1-step TD in warmup)."""
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

        env = share_obs.shape[1]
        uniform_r = (R / N).expand(-1, -1, N)

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
        tpdv = dict(dtype=torch.float32, device=self.device)
        import numpy as np
        if isinstance(share_obs, np.ndarray):
            share_obs = check(share_obs).to(**tpdv)
        if isinstance(rnn_states, np.ndarray):
            rnn_states = check(rnn_states).to(**tpdv)
        if isinstance(masks, np.ndarray):
            masks = check(masks).to(**tpdv)

        T, env = share_obs.shape[0], share_obs.shape[1]
        so_flat = share_obs.reshape(T * env, -1)
        rs_flat = rnn_states.reshape(T * env, *rnn_states.shape[2:])
        m_flat = masks.reshape(T * env, -1)

        values, _ = self.policy.critic(so_flat, rs_flat, m_flat)
        return values.reshape(T, env)
