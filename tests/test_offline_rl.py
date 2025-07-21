import numpy as np

from gym_anm import (
    IEEE33Env,
    generate_dataset,
    generate_mixed_dataset,
    behavior_cloning,
    evaluate_policy,
    SimpleCapBankExpert,
    ConservativeCapBankExpert,
    AggressiveCapBankExpert,
    NoisyCapBankExpert,
    DelayedCapBankExpert,
    LaggingCapBankExpert,
    HysteresisCapBankExpert,
)


def test_offline_rl_basic():

    env = IEEE33Env()
    rand_states, rand_actions = generate_dataset(env, None, 2)
    expert = SimpleCapBankExpert(env)
    exp_states, exp_actions = generate_dataset(env, expert, 2)

    rand_policy = behavior_cloning(rand_states, rand_actions, env.action_space)
    exp_policy = behavior_cloning(exp_states, exp_actions, env.action_space)

    rand_perf = evaluate_policy(env, rand_policy, episodes=1, max_steps=2)
    exp_perf = evaluate_policy(env, exp_policy, episodes=1, max_steps=2)

    assert exp_perf >= rand_perf


def test_mixed_dataset_generation():

    env = IEEE33Env()
    agents = [
        None,
        SimpleCapBankExpert(env),
        ConservativeCapBankExpert(env),
        AggressiveCapBankExpert(env),
        NoisyCapBankExpert(env),
        DelayedCapBankExpert(env),
        LaggingCapBankExpert(env),
    ]

    states, actions = generate_mixed_dataset(env, agents, 5)

    assert states.shape[0] == actions.shape[0] == 5


def test_mixed_dataset_weights():

    env = IEEE33Env()
    expert = SimpleCapBankExpert(env)
    others = [AggressiveCapBankExpert(env)]

    env.reset(seed=42)
    states_a, actions_a = generate_dataset(env, expert, 3)

    env.reset(seed=42)
    states_b, actions_b = generate_mixed_dataset(
        env,
        [expert] + others,
        3,
        weights=[1.0, 0.0],
    )

    np.testing.assert_allclose(actions_a, actions_b)


def test_hysteresis_expert_dataset():
    env = IEEE33Env()
    expert = HysteresisCapBankExpert(env)
    states, actions = generate_dataset(env, expert, 3)
    assert states.shape[0] == actions.shape[0] == 3
