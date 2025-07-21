import unittest
from gym_anm import IEEE33Env
from gym_anm.offline import collect_data, BehaviorCloningPolicy, evaluate_policy
import numpy as np


class TestOfflineRL(unittest.TestCase):
    def test_behavior_cloning(self):
        env = IEEE33Env()
        random_policy = lambda env, obs: env.action_space.sample()
        expert_policy = lambda env, obs: np.array([1.0])

        rand_buffer = collect_data(env, random_policy, episodes=2, max_steps=5)
        expert_buffer = collect_data(env, expert_policy, episodes=2, max_steps=5)

        rand_agent = BehaviorCloningPolicy(env.action_space)
        expert_agent = BehaviorCloningPolicy(env.action_space)
        rand_agent.fit(rand_buffer)
        expert_agent.fit(expert_buffer)

        rand_action = rand_agent.act(env, env.reset()[0])
        expert_action = expert_agent.act(env, env.reset()[0])

        self.assertNotAlmostEqual(rand_action[0], expert_action[0])


if __name__ == "__main__":
    unittest.main()
