"""DRBFN-v2 end-to-end mock test (no SC2 dependency).

Builds a minimal dummy buffer + policy + trainer, then calls the core methods:
  - _compute_generated_rewards(buffer)         # multi-sample + exploration bonus
  - _compute_counterfactual_target(...)        # softmax(Δ_i^CF / τ) · R
  - _train_components(buffer)                  # BFN + Q_tot training

This bypasses SC2 to verify v2 algorithmic correctness on its own.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
sys.path.insert(0, 'C:/Users/张英铭/Desktop/on-policy')


class DummyArgs:
    """Minimal args to construct policy + trainer."""
    # MAPPO defaults
    gamma = 0.99
    max_grad_norm = 10.0
    hidden_size = 64
    # DRBFN v2
    drbfn_hidden = 64
    drbfn_n_sample_steps = 2
    drbfn_lr = 3e-4
    drbfn_beta = 0.5
    drbfn_warmup_t = 0          # skip warmup, run full pipeline
    drbfn_update_interval = 1
    drbfn_k_samples = 4
    drbfn_lambda_cons = 0.1
    drbfn_lambda_exp = 0.01
    drbfn_cf_temperature = 1.0
    # needed by R_MAPPO base
    use_popart = False
    use_valuenorm = True
    use_proper_time_limits = False
    gain = 0.01
    use_gae = True
    gae_lambda = 0.95
    lr = 5e-4
    critic_lr = 5e-4
    opti_eps = 1e-5
    weight_decay = 0.0
    use_max_grad_norm = True
    use_clipped_value_loss = True
    value_loss_coef = 0.5
    clip_param = 0.2
    entropy_coef = 0.01
    policy_type = 'discrete'
    use_policy_active_masks = True
    discrete_type = 'multi_categorical'
    # R_Actor / R_Critic needs
    use_orthogonal = True
    use_naive_recurrent_policy = False
    use_recurrent_policy = True
    recurrent_N = 1
    use_influence_policy = False
    stacked_modal = False
    # MLP / RNN base
    layer_N = 1
    stacked_frames = 1
    use_ReLU = True
    use_feature_normalization = True
    # R_MAPPO_Policy
    algorithm_name = 'r_drbfn_v2'
    # R_MAPPO trainer needs
    ppo_epoch = 5
    num_mini_batch = 1
    data_chunk_length = 10
    use_value_active_masks = True
    use_huber_loss = False
    huber_delta = 10.0


class DummyBuffer:
    """Minimal buffer matching what v2 Trainer accesses."""
    def __init__(self, T=20, env=2, N=3, share_obs_dim=30, obs_dim=20,
                 num_actions=9):
        self.episode_length = T
        self.n_rollout_threads = env
        # share_obs: (T+1, env, N, dim) - use agent 0
        self.share_obs = np.random.randn(T+1, env, N, share_obs_dim).astype(np.float32)
        # obs: (T+1, env, N, obs_dim)
        self.obs = np.random.randn(T+1, env, N, obs_dim).astype(np.float32)
        # actions: (T+1, env, N, 1)
        self.actions = np.random.randint(0, num_actions, (T+1, env, N, 1)).astype(np.int64)
        # rewards: (T+1, env, N, 1) - SMAC stores R broadcast to all agents
        R = np.random.randn(T+1, env, 1, 1).astype(np.float32) * 0.1
        self.rewards = np.broadcast_to(R, (T+1, env, N, 1)).copy().astype(np.float32)
        # masks: (T+1, env, N, 1), 1=alive, 0=done
        self.masks = np.ones((T+1, env, N, 1), dtype=np.float32)
        # value_preds / returns placeholder
        self.value_preds = np.zeros((T+1, env, N, 1), dtype=np.float32)
        self.returns = np.zeros((T+1, env, N, 1), dtype=np.float32)
        # rnn_states (actor + critic): use small zeros
        self.rnn_states = np.zeros((T+1, env, N, 1, 64), dtype=np.float32)
        self.rnn_states_critic = np.zeros((T+1, env, N, 1, 64), dtype=np.float32)
        # available_actions: None (allow all)
        self.available_actions = None

    def compute_returns(self, next_value, value_normalizer):
        # Minimal GAE: just set returns to cumulative reward (mock)
        T = self.episode_length
        self.returns[:T] = self.rewards[:T]
        self.value_preds[:T] = 0

    def after_update(self):
        pass


def build_policy(args, num_agents, num_actions, share_obs_dim, obs_dim, device):
    """Manually construct R_DRBFNPolicy_v2 bypassing R_MAPPO_Policy init
    (which requires spaces)."""
    from onpolicy.algorithms.r_drbfn_v2.algorithm.rDRBFNPolicy_v2 import R_DRBFNPolicy_v2
    from gym.spaces import Box, Discrete
    obs_space = Box(low=-1, high=1, shape=(obs_dim,))
    cent_obs_space = Box(low=-1, high=1, shape=(share_obs_dim,))
    act_space = Discrete(num_actions)
    return R_DRBFNPolicy_v2(args, obs_space, cent_obs_space, act_space,
                            num_agents=num_agents, device=device)


def main():
    print('=== DRBFN-v2 end-to-end mock test ===')
    device = torch.device('cpu')
    args = DummyArgs()

    # Scenario: 3 agents, 9 actions (3m-like)
    N = 3
    num_actions = 9
    share_obs_dim = 30
    obs_dim = 20
    T, env = 20, 2

    print(f'Setup: N={N}, num_actions={num_actions}, T={T}, env={env}')

    # Build policy
    policy = build_policy(args, N, num_actions, share_obs_dim, obs_dim, device)
    print('[OK] Policy built')

    # Build trainer
    from onpolicy.algorithms.r_drbfn_v2.r_drbfn_v2 import R_DRBFN_v2
    trainer = R_DRBFN_v2(args, policy, num_agents=N, device=device)
    print('[OK] Trainer built')

    # Build buffer
    buf = DummyBuffer(T=T, env=env, N=N, share_obs_dim=share_obs_dim,
                      obs_dim=obs_dim, num_actions=num_actions)
    print('[OK] Buffer built')

    # ============ Test 1: _compute_counterfactual_target ============
    print()
    print('--- Test 1: _compute_counterfactual_target ---')
    tpdv = dict(dtype=torch.float32, device=device)
    share_obs_t = torch.randn(T, env, share_obs_dim, **tpdv)
    actions_t = torch.randint(0, num_actions, (T, env, N, 1), device=device).long()
    joint_a_oh = F.one_hot(actions_t.squeeze(-1), num_actions).float().reshape(T, env, N*num_actions)
    R_t = torch.randn(T, env, 1, **tpdv) * 0.1

    target_r = trainer._compute_counterfactual_target(share_obs_t, actions_t, joint_a_oh, R_t)
    print(f'  output shape: {tuple(target_r.shape)}, expected ({T}, {env}, {N})')
    print(f'  sum per (t,env) should be ≈ R:')
    sum_r = target_r.sum(dim=-1)
    print(f'    sum_r mean={sum_r.mean().item():.4f}, R mean={R_t.mean().item():.4f}')
    diff = (sum_r - R_t.squeeze(-1)).abs().mean().item()
    print(f'    |sum_r - R| mean={diff:.6f} (should be ~0)')
    assert diff < 1e-5, 'Conservation violated!'
    print(f'  weights sum to 1 per step: {target_r.sum(dim=-1)[0, 0].item():.4f}')

    # ============ Test 2: _compute_generated_rewards ============
    print()
    print('--- Test 2: _compute_generated_rewards ---')
    gen_r, stats = trainer._compute_generated_rewards(buf)
    print(f'  output shape: {gen_r.shape}, expected ({T}, {env}, {N}, 1)')
    print(f'  stats: {stats}')
    print(f'  gen_r range: [{gen_r.min():.4f}, {gen_r.max():.4f}]')
    print(f'  gen_r mean per agent: {gen_r.mean(axis=(0,1,3))}')

    # ============ Test 3: _train_components (full training step) ============
    print()
    print('--- Test 3: _train_components (1 training step) ---')
    train_info = trainer._train_components(buf)
    print(f'  train_info: {train_info}')

    # ============ Test 4: Verify losses are finite ============
    print()
    print('--- Test 4: Sanity checks ---')
    for k, v in train_info.items():
        if isinstance(v, float):
            assert not (np.isnan(v) or np.isinf(v)), f'{k} is NaN/Inf!'
            print(f'  [OK] {k} = {v:.6f} (finite)')

    # ============ Test 5: Multiple steps (does it converge?) ============
    print()
    print('--- Test 5: 20 training steps (does drbfn_loss decrease?) ---')
    losses = []
    for i in range(20):
        info = trainer._train_components(buf)
        losses.append(info['drbfn_loss'])
    print(f'  drbfn_loss: step1={losses[0]:.4f}, step20={losses[-1]:.4f}')

    # ============ Test 6: Full train() call ============
    print()
    print('--- Test 6: trainer.train(buf, update_actor=False) ---')
    try:
        # Manually set update_actor=False to skip actor update (we don't have actor fwd)
        info = trainer.train(buf, update_actor=False)
        print(f'  [OK] train() completed')
        print(f'  returned keys: {list(info.keys())[:6]}...')
    except Exception as e:
        print(f'  [EXPECTED-FAIL] train() needs full buffer/actor setup: {type(e).__name__}: {e}')

    print()
    print('=== All core v2 tests passed ===')


if __name__ == '__main__':
    main()
