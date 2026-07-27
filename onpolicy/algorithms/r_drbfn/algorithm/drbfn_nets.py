"""DRBFN networks: Q_i (per-agent) and Q_tot (global), reusing BFN from utils/drbfn.py."""
import torch
import torch.nn as nn


class IndividualQ(nn.Module):
    """Per-agent Q_i network (MLP, CTDE).
    Input: [share_obs, obs_i, agent_id_onehot] -> Q_i(s, a) for all actions.
    """
    def __init__(self, share_obs_dim, obs_dim, num_agents, num_actions, hidden_size):
        super().__init__()
        input_dim = share_obs_dim + obs_dim + num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions),
        )
        self.share_obs_dim = share_obs_dim
        self.obs_dim = obs_dim
        self.num_agents = num_agents

    def forward(self, share_obs, obs, agent_id_onehot):
        # share_obs: (..., share_obs_dim), obs: (..., num_agents, obs_dim)
        # agent_id_onehot: (..., num_agents, num_agents)
        so = share_obs.unsqueeze(-2).expand(*share_obs.shape[:-1], self.num_agents, self.share_obs_dim)
        inp = torch.cat([so, obs, agent_id_onehot], dim=-1)
        return self.net(inp)  # (..., num_agents, num_actions)


class GlobalQ(nn.Module):
    """Independent global Q_tot network (MLP).
    Input: [share_obs, joint_action_onehot] -> scalar Q_tot.
    """
    def __init__(self, share_obs_dim, num_agents, num_actions, hidden_size):
        super().__init__()
        input_dim = share_obs_dim + num_agents * num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, share_obs, joint_action_onehot):
        inp = torch.cat([share_obs, joint_action_onehot], dim=-1)
        return self.net(inp)  # (..., 1)
