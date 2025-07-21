import numpy as np
from typing import Optional, Callable, Sequence
from .envs.ieee33_env import IEEE33Env
from .simulator.components import CapacitorBank


def generate_dataset(env, agent: Optional[Callable], steps: int):
    """Collect a dataset of (state, action) pairs."""
    states, actions = [], []
    obs, _ = env.reset()
    for _ in range(steps):
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.act(env)
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return np.array(states), np.array(actions)


def generate_mixed_dataset(env, agents: Sequence[Optional[Callable]], steps: int):
    """Collect a dataset from a mixture of agents.

    Parameters
    ----------
    env : gym.Env
        Environment in which to collect data.
    agents : sequence of callables or ``None``
        Agents used to generate the actions. If an element is ``None`` a random
        action is sampled.
    steps : int
        Number of environment steps to record.

    Returns
    -------
    states : :class:`numpy.ndarray`
        Recorded observations.
    actions : :class:`numpy.ndarray`
        Actions taken at each state.
    """

    states, actions = [], []
    obs, _ = env.reset()
    for _ in range(steps):
        agent = np.random.choice(agents)
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.act(env)
        next_obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs)
        actions.append(action)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    return np.array(states), np.array(actions)


def behavior_cloning(states: np.ndarray, actions: np.ndarray, action_space):
    X = np.concatenate([states, np.ones((states.shape[0], 1))], axis=1)
    w, _, _, _ = np.linalg.lstsq(X, actions, rcond=None)

    def policy(state):
        a = np.dot(np.append(state, 1.0), w)
        return np.clip(a, action_space.low, action_space.high)

    return policy


def evaluate_policy(env, policy, episodes: int = 1, max_steps: int = 10):
    total_reward = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            action = policy(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            steps += 1
    return total_reward / episodes


class CapBankExpert:
    """General capacitor bank heuristic with configurable thresholds."""

    def __init__(self, env: IEEE33Env, v_min: float = 0.99, v_max: float = 1.01):
        self.env = env
        self.v_min = v_min
        self.v_max = v_max
        self.cap_ids = [
            i
            for i, d in env.unwrapped.simulator.devices.items()
            if isinstance(d, CapacitorBank)
        ]

    def act(self, env: IEEE33Env):
        action = np.zeros(env.action_space.shape[0])
        sim = env.unwrapped.simulator
        base = 0
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = np.abs(sim.buses[dev.bus_id].v)
            if bus_v < self.v_min:
                q = dev.q_max * sim.baseMVA
            elif bus_v > self.v_max:
                q = dev.q_min * sim.baseMVA
            else:
                q = 0.0
            action[base + idx] = q
        return action


class SimpleCapBankExpert(CapBankExpert):
    """Heuristic expert with 0.99/1.01 voltage thresholds."""

    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.99, v_max=1.01)


class ConservativeCapBankExpert(CapBankExpert):
    """Expert that acts only for larger voltage deviations."""

    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.98, v_max=1.02)


class AggressiveCapBankExpert(CapBankExpert):
    """Expert that reacts to small voltage deviations."""

    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.995, v_max=1.005)


class NoisyCapBankExpert(CapBankExpert):
    """Expert that senses voltages with Gaussian noise."""

    def __init__(self, env: IEEE33Env, noise_std: float = 0.005):
        super().__init__(env)
        self.noise_std = noise_std

    def act(self, env: IEEE33Env):
        action = np.zeros(env.action_space.shape[0])
        sim = env.unwrapped.simulator
        base = 0
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = np.abs(sim.buses[dev.bus_id].v)
            bus_v += np.random.normal(0.0, self.noise_std)
            if bus_v < self.v_min:
                q = dev.q_max * sim.baseMVA
            elif bus_v > self.v_max:
                q = dev.q_min * sim.baseMVA
            else:
                q = 0.0
            action[base + idx] = q
        return action


class DelayedCapBankExpert(CapBankExpert):
    """Expert that only updates its action every ``delay`` steps."""

    def __init__(self, env: IEEE33Env, delay: int = 2):
        super().__init__(env)
        self.delay = max(1, delay)
        self._counter = 0

    def act(self, env: IEEE33Env):
        if self._counter % self.delay != 0:
            self._counter += 1
            return np.zeros(env.action_space.shape[0])
        self._counter += 1
        return super().act(env)
