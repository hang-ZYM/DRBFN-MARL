"""R_DRBFN_QVPO Policy.

Networks (4 + 1 target):
  - Actor π_i (from R_MAPPO base)
  - V-critic (from R_MAPPO base, for PPO GAE baseline)
  - Q_tot(s, a) — team Q trained on R, also used as BFN signal source
  - BFN (PotentialBFN, outputs Φ ∈ ℝ^N, action-aware)
  - Q_tot target (soft update)

Key difference from v_final:
  - BFN outputs Φ (potential), not r (stick-breaking)
  - r is computed externally via PBRS: r_i = R/N + γΦ_i(s', a') - Φ_i(s, a)
  - No Q_i network
"""
import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_drbfn_qvpo.algorithm.drbfn_qvpo import PotentialBFN
from onpolicy.algorithms.r_drbfn.algorithm.drbfn_nets import GlobalQ


class R_DRBFNQVPOPolicy(R_MAPPOPolicy):
    """v_qvpo Policy: BFN outputs Φ, r via PBRS, no Q_i."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents,
                 device=torch.device("cpu")):
        super().__init__(args, obs_space, cent_obs_space, act_space, device)

        self.num_agents = num_agents
        self.num_actions = act_space.n if hasattr(act_space, 'n') else act_space[0].n
        self.hidden_size = args.hidden_size

        from onpolicy.utils.util import get_shape_from_obs_space
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.share_obs_dim = share_obs_shape[0]
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[0]

        # BFN input: (s, a_joint)
        drbfn_raw_dim = self.share_obs_dim + self.num_agents * self.num_actions

        # BFN: outputs per-agent potential Φ ∈ ℝ^N
        self.drbfn = PotentialBFN(
            raw_dim=drbfn_raw_dim,
            n_agents=num_agents,
            hidden=args.drbfn_hidden,
            n_sample_steps=args.drbfn_n_sample_steps,
        )

        # Q_tot: team Q trained on R
        self.qtot_net = GlobalQ(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target = type(self.qtot_net)(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target.load_state_dict(self.qtot_net.state_dict())
        for p in self.qtot_target.parameters():
            p.requires_grad = False

        self.drbfn.to(device)
        self.qtot_net.to(device)
        self.qtot_target.to(device)

        self.drbfn_optimizer = torch.optim.Adam(self.drbfn.parameters(),
                                                lr=args.drbfn_lr)
        self.qtot_optimizer = torch.optim.Adam(self.qtot_net.parameters(),
                                               lr=args.drbfn_lr)
        # Save initial LRs for lr_decay
        self.drbfn_lr = args.drbfn_lr

        import torch as _t
        self.agent_id_eye = _t.eye(num_agents, device=device)

        self.tau_targets = 0.005
        self.warmup_t = args.drbfn_warmup_t
        self.drbfn_update_interval = args.drbfn_update_interval
        self.n_step = getattr(args, 'drbfn_n_step', 5)
        # K-sample sizes
        self.K_train = getattr(args, 'drbfn_K_train', 4)
        self.K_deploy = getattr(args, 'drbfn_K_deploy', 8)
        # Φ clamp magnitude (prevents r from drifting too far)
        self.phi_clamp = getattr(args, 'drbfn_phi_clamp', 10.0)
        # Default action for counterfactual g_i (SMAC: no-op = 0)
        self.default_action = getattr(args, 'drbfn_default_action', 0)

    def soft_update_qtot(self):
        for tp, p in zip(self.qtot_target.parameters(), self.qtot_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)

    def lr_decay(self, episode, episodes):
        """Decay learning rates for ALL networks (actor, critic, BFN, Q_tot)."""
        # Base MAPPO: decay actor and critic
        from onpolicy.utils.util import update_linear_schedule
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        # QVPO additions: also decay BFN and Q_tot
        update_linear_schedule(self.drbfn_optimizer, episode, episodes, self.drbfn_lr)
        update_linear_schedule(self.qtot_optimizer, episode, episodes, self.drbfn_lr)
