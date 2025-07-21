import numpy as np
from typing import Optional, Callable
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


class SimpleCapBankExpert:
    """A heuristic expert that tries to keep voltages close to 1.0 pu."""

    def __init__(self, env: IEEE33Env):
        self.env = env
        self.cap_ids = [
            i
            for i, d in env.unwrapped.simulator.devices.items()
            if isinstance(d, CapacitorBank)
        ]

    def act(self, env: IEEE33Env):
        action = np.zeros(env.action_space.shape[0])
        sim = env.unwrapped.simulator
        base = 0
        # only capacitor actions present
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = np.abs(sim.buses[dev.bus_id].v)
            if bus_v < 0.99:
                q = dev.q_max * sim.baseMVA
            elif bus_v > 1.01:
                q = dev.q_min * sim.baseMVA
            else:
                q = 0.0
            action[base + idx] = q
        return action
