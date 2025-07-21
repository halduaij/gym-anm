import numpy as np
from ..anm_env import ANMEnv
from .network import network


class IEEE33Env(ANMEnv):
    """ANM environment using the IEEE 33-bus distribution system."""

    metadata = {"render_modes": []}

    def __init__(self):
        observation = "state"
        K = 0
        delta_t = 1.0
        gamma = 0.99
        lamb = 100
        super().__init__(network, observation, K, delta_t, gamma, lamb)

    def init_state(self):
        n_dev = self.simulator.N_device
        n_des = self.simulator.N_des
        n_gen = self.simulator.N_non_slack_gen
        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        # initialize loads to their nominal demand
        idx = 0
        for dev_id, dev in self.simulator.devices.items():
            if dev.is_slack:
                continue
            p = dev.p_min
            if hasattr(dev, "qp_ratio") and dev.qp_ratio is not None:
                q = p * dev.qp_ratio
            else:
                q = 0.0
            state[dev_id] = p
            state[n_dev + dev_id] = q
        return state

    def next_vars(self, s_t):
        # no stochastic variation
        return np.zeros(self.simulator.N_load + self.simulator.N_non_slack_gen + self.K)
