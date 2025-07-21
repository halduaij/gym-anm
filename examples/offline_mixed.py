"""Example of collecting a dataset from multiple agents."""

import gymnasium as gym
from gym_anm import (
    generate_mixed_dataset,
    SimpleCapBankExpert,
    ConservativeCapBankExpert,
    AggressiveCapBankExpert,
    NoisyCapBankExpert,
    DelayedCapBankExpert,
)

# BEGIN OFFLINE MIXED EXAMPLE
env = gym.make("gym_anm:IEEE33-v0")
expert1 = SimpleCapBankExpert(env)
expert2 = ConservativeCapBankExpert(env)
expert3 = AggressiveCapBankExpert(env)
expert4 = NoisyCapBankExpert(env)
expert5 = DelayedCapBankExpert(env)
agents = [None, expert1, expert2, expert3, expert4, expert5]
weights = [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]

states, actions = generate_mixed_dataset(env, agents, steps=10, weights=weights)
# END OFFLINE MIXED EXAMPLE
