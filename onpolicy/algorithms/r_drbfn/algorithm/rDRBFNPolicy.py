"""R_DRBFN Policy: extends R_MAPPO_Policy with BFN, Q_i, Q_tot networks."""
import torch
import torch.nn as nn
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from onpolicy.algorithms.utils.drbfn import UnifiedWorldModel
from onpolicy.algorithms.r_drbfn.algorithm.drbfn_nets import IndividualQ, GlobalQ


class R_DRBFNPolicy(R_MAPPOPolicy):
    """MAPPO Policy + DRBFN components (BFN + Q_i + Q_tot)."""

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

        # DRBFN raw input dim: [share_obs, joint_action_onehot]
        drbfn_raw_dim = self.share_obs_dim + self.num_agents * self.num_actions

        # DRBFN BFN module
        self.drbfn = UnifiedWorldModel(
            raw_dim=drbfn_raw_dim,
            n_actions=self.num_actions,
            n_agents=num_agents,
            hidden=args.drbfn_hidden,
            n_sample_steps=args.drbfn_n_sample_steps,
        )

        # Per-agent Q_i
        self.qi_net = IndividualQ(
            self.share_obs_dim, self.obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qi_target = type(self.qi_net)(
            self.share_obs_dim, self.obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qi_target.load_state_dict(self.qi_net.state_dict())
        for p in self.qi_target.parameters():
            p.requires_grad = False

        # Global Q_tot
        self.qtot_net = GlobalQ(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target = type(self.qtot_net)(
            self.share_obs_dim, num_agents, self.num_actions, args.hidden_size
        )
        self.qtot_target.load_state_dict(self.qtot_net.state_dict())
        for p in self.qtot_target.parameters():
            p.requires_grad = False

        # Move all to device
        self.drbfn.to(device)
        self.qi_net.to(device)
        self.qi_target.to(device)
        self.qtot_net.to(device)
        self.qtot_target.to(device)

        # Additional optimizers
        self.drbfn_optimizer = torch.optim.Adam(self.drbfn.parameters(),
                                                lr=args.drbfn_lr)
        self.qi_optimizer = torch.optim.Adam(self.qi_net.parameters(),
                                             lr=args.qi_lr)
        self.qtot_optimizer = torch.optim.Adam(self.qtot_net.parameters(),
                                               lr=args.drbfn_lr)

        # Agent ID onehot (buffer)
        import torch as _t
        self.agent_id_eye = _t.eye(num_agents, device=device)

        # DRBFN hyperparams
        self.drbfn_beta = args.drbfn_beta
        self.tau_targets = 0.005
        self.warmup_t = args.drbfn_warmup_t
        self.drbfn_update_interval = args.drbfn_update_interval

    def soft_update_qi(self):
        for tp, p in zip(self.qi_target.parameters(), self.qi_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)

    def soft_update_qtot(self):
        for tp, p in zip(self.qtot_target.parameters(), self.qtot_net.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau_targets) + p.data * self.tau_targets)
