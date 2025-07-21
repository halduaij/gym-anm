import numpy as np
from sklearn.neural_network import MLPRegressor

from .dataset import DataBuffer


def collect_data(env, policy, episodes=1, max_steps=50):
    """Collect transitions by running `policy` in `env`."""
    buffer = DataBuffer()
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            if hasattr(policy, "act"):
                action = policy.act(env) if callable(getattr(policy, "act")) else policy(env, obs)
            else:
                action = policy(env, obs)
            next_obs, reward, done, _, _ = env.step(action)
            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            if done:
                break
    return buffer


class BehaviorCloningPolicy:
    """Simple behavior cloning using MLPRegressor."""

    def __init__(self, action_space):
        self.action_space = action_space
        self.model = MLPRegressor(hidden_layer_sizes=(32, 32), max_iter=500)

    def fit(self, buffer: DataBuffer):
        states, actions, *_ = buffer.arrays()
        self.model.fit(states, actions)

    def act(self, env, obs=None):
        if obs is None:
            obs = env.state
        a = self.model.predict(np.asarray(obs).reshape(1, -1))[0]
        return np.clip(a, self.action_space.low, self.action_space.high)


def evaluate_policy(env, policy, episodes=1, max_steps=50):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        for _ in range(max_steps):
            if hasattr(policy, "act"):
                action = policy.act(env, obs)
            else:
                action = policy(env, obs)
            obs, reward, done, _, _ = env.step(action)
            total += reward
            if done:
                break
    return total / episodes
