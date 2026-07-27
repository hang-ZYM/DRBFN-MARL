"""DRBFN-v2 networks: only GlobalQ remains (IndividualQ removed).

In v2 the BFN directly generates per-agent rewards r_i, so we no longer
maintain a separate Q_i network. Q_tot's role changes from "decomposition
target" to "counterfactual inference tool": we use it to compute
    r_i^target = Q_tot(s, a) - Q_tot(s, a_{-i}, a_i')
which is the counterfactual baseline (COMA-style).
"""
import torch
import torch.nn as nn


class GlobalQ(nn.Module):
    """Independent global Q_tot network (MLP).
    Input: [share_obs, joint_action_onehot] -> scalar Q_tot.

    Used for counterfactual inference: r_i^CF = Q_tot(s,a) - Q_tot(s, a_{-i}, a_i').
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
        self.num_agents = num_agents
        self.num_actions = num_actions

    def forward(self, share_obs, joint_action_onehot):
        """share_obs: (..., share_obs_dim)
        joint_action_onehot: (..., num_agents * num_actions)
        Returns: (..., 1)
        """
        inp = torch.cat([share_obs, joint_action_onehot], dim=-1)
        return self.net(inp)
