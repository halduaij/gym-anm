import unittest
import numpy as np

from gym_anm import IEEE33Env


class TestIEEE33Env(unittest.TestCase):
    def test_reset_and_step(self):
        env = IEEE33Env()
        obs, info = env.reset()
        self.assertEqual(obs.shape[0], env.observation_space.shape[0])
        a = env.action_space.sample()
        obs, r, terminated, _, _ = env.step(a)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertIsInstance(r, float)
        self.assertIsInstance(terminated, bool)


if __name__ == "__main__":
    unittest.main()
