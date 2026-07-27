"""R_DRBFN-v3 Policy: same networks as v1 (BFN + Q_i + Q_tot).
v3 改动只在 Trainer：n-step return for Q_tot + n-step counterfactual for BFN.
"""
import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.utils.drbfn import UnifiedWorldModel
from onpolicy.algorithms.r_drbfn.algorithm.drbfn_nets import IndividualQ, GlobalQ


class R_DRBFNPolicy_v3(R_MAPPOPolicy):
    """Same architecture as v1: MAPPO + BFN + Q_i + Q_tot + Dual Gate.
    Trainer-level changes (n-step) live in r_drbfn_v3.py.
    """

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

        drbfn_raw_dim = self.share_obs_dim + self.num_agents * self.num_actions

        self.drbfn = UnifiedWorldModel(
            raw_dim=drbfn_raw_dim,
            n_actions=self.num_actions,
            n_agents=num_agents,
            hidden=args.drbfn_hidden,
            n_sample_steps=args.drbfn_n_sample_steps,
        )

        self.qi_net = IndividualQ(
            self.share_obs_dim, self.obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qi_target = type(self.qi_net)(
            self.share_obs_dim, self.obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qi_target.load_state_dict(self.qi_net.state_dict())
        for p in self.qi_target.parameters():
            p.requires_grad = False

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
        self.qi_net.to(device)
        self.qi_target.to(device)
        self.qtot_net.to(device)
        self.qtot_target.to(device)

        self.drbfn_optimizer = torch.optim.Adam(self.drbfn.parameters(),
                                                lr=args.drbfn_lr)
        self.qi_optimizer = torch.optim.Adam(self.qi_net.parameters(),
                                             lr=args.qi_lr)
        self.qtot_optimizer = torch.optim.Adam(self.qtot_net.parameters(),
                                               lr=args.drbfn_lr)

        import torch as _t
        self.agent_id_eye = _t.eye(num_agents, device=device)

        self.drbfn_beta = args.drbfn_beta
        self.tau_targets = 0.005
        self.warmup_t = args.drbfn_warmup_t
        self.drbfn_update_interval = args.drbfn_update_interval
        # v3 new: n-step return horizon
        self.n_step = getattr(args, 'drbfn_n_step', 5)

    def soft_update_qi(self):
        for tp, p in zip(self.qi_target.parameters(), self.qi_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)

    def soft_update_qtot(self):
        for tp, p in zip(self.qtot_target.parameters(), self.qtot_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)
