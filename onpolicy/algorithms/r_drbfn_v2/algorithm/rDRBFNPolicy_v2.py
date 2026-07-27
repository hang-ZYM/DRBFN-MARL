"""R_DRBFN-v2 Policy: generative per-agent reward via BFN.

Architecture (3 networks + actor/critic):
  - RewardGenerator (BFN):    generates r ∈ ℝ^N from (s, a, R)
  - GlobalQ (Q_tot):          counterfactual inference (r_i^CF = Q_tot(s,a) - Q_tot(s, a_{-i}, a_i'))
  - Actor / Critic:           inherited from R_MAPPO (standard MAPPO)

Removed from v1:
  - IndividualQ (Q_i):        BFN now directly generates r_i, no per-agent Q needed
  - Dual Gate (δQ-based):     replaced by variance-based confidence from BFN

The Dual Gate used the old Q_i (A_DRBFN) and Q_tot (δQ), which created a circular
dependency (r_i trains Q_i, Q_i decides whether r_i is used). v2 drops this gate.
"""
import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.utils.drbfn_v2 import RewardGenerator
from onpolicy.algorithms.r_drbfn_v2.algorithm.drbfn_nets_v2 import GlobalQ


class R_DRBFNPolicy_v2(R_MAPPOPolicy):
    """MAPPO Policy + DRBFN-v2 generative reward components."""

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents,
                 device=torch.device("cpu")):
        super().__init__(args, obs_space, cent_obs_space, act_space, device)

        self.num_agents = num_agents
        self.num_actions = act_space.n if hasattr(act_space, 'n') else act_space[0].n
        self.hidden_size = args.hidden_size

        # Infer input dims
        from onpolicy.utils.util import get_shape_from_obs_space
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)
        self.share_obs_dim = share_obs_shape[0]
        obs_shape = get_shape_from_obs_space(obs_space)
        self.obs_dim = obs_shape[0]

        # v2 raw input: [share_obs, joint_action_onehot, R]
        # R is added as a conditioning feature (so generator can scale r_i with R)
        drbfn_raw_dim = self.share_obs_dim + self.num_agents * self.num_actions + 1

        # BFN reward generator
        self.drbfn = RewardGenerator(
            raw_dim=drbfn_raw_dim,
            n_actions=self.num_actions,
            n_agents=num_agents,
            hidden=args.drbfn_hidden,
            n_sample_steps=args.drbfn_n_sample_steps,
        )

        # Global Q_tot (for counterfactual inference)
        self.qtot_net = GlobalQ(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target = type(self.qtot_net)(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target.load_state_dict(self.qtot_net.state_dict())
        for p in self.qtot_target.parameters():
            p.requires_grad = False

        # Move to device
        self.drbfn.to(device)
        self.qtot_net.to(device)
        self.qtot_target.to(device)

        # Optimizers
        self.drbfn_optimizer = torch.optim.Adam(self.drbfn.parameters(),
                                                lr=args.drbfn_lr)
        self.qtot_optimizer = torch.optim.Adam(self.qtot_net.parameters(),
                                               lr=args.drbfn_lr)

        # Hyperparams
        self.tau_targets = 0.005
        self.warmup_t = args.drbfn_warmup_t
        self.drbfn_update_interval = args.drbfn_update_interval
        self.cf_temperature = getattr(args, 'drbfn_cf_temperature', 1.0)

    def soft_update_qtot(self):
        for tp, p in zip(self.qtot_target.parameters(), self.qtot_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)
