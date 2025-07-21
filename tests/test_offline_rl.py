import numpy as np

from gym_anm import (
    IEEE33Env,
    generate_dataset,
    behavior_cloning,
    evaluate_policy,
    SimpleCapBankExpert,
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