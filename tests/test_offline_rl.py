import numpy as np
from gym_anm import generate_dataset, behavior_cloning, evaluate_policy, ANMEnv
from gym_anm.simulator import Simulator
import numpy as np


class ZeroAgent:
    def act(self, env):
        return np.zeros(env.action_space.shape[0])


class MiniEnv(ANMEnv):
    def __init__(self):
        network = {
            "baseMVA": 1,
            "bus": np.array([[0, 0, 50, 1.0, 1.0], [1, 1, 50, 1.1, 0.9]]),
            "branch": np.array([[0, 1, 0.01, 0.1, 0.0, 32, 1, 0]]),
            "device": np.array(
                [
                    [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
                    [1, 1, 4, None, 0, 0, 1, -1, None, None, None, None, None, None, None],
                ]
            ),
        }
        super().__init__(network, "state", 0, 1.0, 0.99, 100)

    def init_state(self):
        n_dev = self.simulator.N_device
        state = np.zeros(2 * n_dev)
        for dev_id, dev in self.simulator.devices.items():
            if not dev.is_slack:
                state[dev_id] = dev.p_min
                state[n_dev + dev_id] = 0.0
        return state

    def next_vars(self, s_t):
        return np.zeros(self.simulator.N_load + self.simulator.N_non_slack_gen)


def test_offline_rl_basic():
    env = MiniEnv()
    rand_states, rand_actions = generate_dataset(env, None, 3)
    expert = ZeroAgent()
    exp_states, exp_actions = generate_dataset(env, expert, 3)

    rand_policy = behavior_cloning(rand_states, rand_actions, env.action_space)
    exp_policy = behavior_cloning(exp_states, exp_actions, env.action_space)

    rand_perf = evaluate_policy(env, rand_policy, episodes=1)
    exp_perf = evaluate_policy(env, exp_policy, episodes=1)

    assert exp_perf >= rand_perf
