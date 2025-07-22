import numpy as np
from typing import Optional, Callable, Sequence
from .envs.ieee33_env import IEEE33Env
from .simulator.components import CapacitorBank, OLTC, Generator, StorageUnit


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


def generate_mixed_dataset(
    env, agents: Sequence[Optional[Callable]], steps: int, weights: Optional[Sequence[float]] = None
):
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
    weights : sequence of float, optional
        Selection probabilities for each agent. If omitted agents are chosen
        uniformly at random.

    Returns
    -------
    states : :class:`numpy.ndarray`
        Recorded observations.
    actions : :class:`numpy.ndarray`
        Actions taken at each state.
    """

    states, actions = [], []
    obs, _ = env.reset()

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(agents):
            raise ValueError("Length of weights must match number of agents")
        w = w / w.sum()

    for _ in range(steps):
        if weights is None:
            idx = np.random.randint(len(agents))
        else:
            idx = int(np.random.choice(len(agents), p=w))
        agent = agents[idx]
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
        sim = env.unwrapped.simulator
        self.cap_ids = [i for i, d in sim.devices.items() if isinstance(d, CapacitorBank)]
        self.oltc_ids = [i for i, d in sim.devices.items() if isinstance(d, OLTC)]
        self.ren_gen_ids = [
            i for i, d in sim.devices.items() if isinstance(d, Generator) and d.type == 2 and not d.is_slack
        ]
        self.des_ids = [i for i, d in sim.devices.items() if isinstance(d, StorageUnit)]

    def act(self, env: IEEE33Env):
        sim = env.unwrapped.simulator
        gen_ids = [i for i, d in sim.devices.items() if isinstance(d, Generator) and not d.is_slack]
        des_ids = [i for i, d in sim.devices.items() if isinstance(d, StorageUnit)]

        cap_ids = self.cap_ids
        oltc_ids = self.oltc_ids

        N_gen = len(gen_ids)
        N_des = len(des_ids)
        N_cap = len(cap_ids)
        action = np.zeros(env.action_space.shape[0])

        base = 0
        for idx, dev_id in enumerate(gen_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = abs(sim.buses[gen.bus_id].v)
                if v > self.v_max:
                    p = max(gen.p_min, 0.9 * gen.p_pot)
                else:
                    p = gen.p_pot
                action[base + idx] = p * sim.baseMVA
            else:
                action[base + idx] = gen.p_pot * sim.baseMVA
        base += N_gen

        for idx in range(N_gen):
            action[base + idx] = 0.0
        base += N_gen

        for idx in range(N_des):
            action[base + idx] = 0.0
        base += N_des
        for idx in range(N_des):
            action[base + idx] = 0.0
        base += N_des

        for idx, dev_id in enumerate(cap_ids):
            action[base + idx] = 0.0
        base += N_cap

        for idx, dev_id in enumerate(oltc_ids):
            action[base + idx] = sim.devices[dev_id].tap

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
        action = CapBankExpert.act(self, env)
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
            return CapBankExpert.act(self, env)
        self._counter += 1
        return CapBankExpert.act(self, env)


class LaggingCapBankExpert(CapBankExpert):
    """Expert using voltage measurements from ``lag`` steps ago."""

    def __init__(self, env: IEEE33Env, lag: int = 1):
        super().__init__(env)
        self.lag = max(1, lag)
        self._history = []

    def act(self, env: IEEE33Env):
        sim = env.unwrapped.simulator
        current_vs = [np.abs(sim.buses[sim.devices[dev_id].bus_id].v) for dev_id in self.cap_ids]
        self._history.append(current_vs)
        if len(self._history) <= self.lag:
            used_vs = current_vs
        else:
            used_vs = self._history[-self.lag - 1]
            self._history = self._history[-self.lag - 1 :]

        action = CapBankExpert.act(self, env)
        base = 0
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = used_vs[idx]
            if bus_v < self.v_min:
                q = dev.q_max * sim.baseMVA
            elif bus_v > self.v_max:
                q = dev.q_min * sim.baseMVA
            else:
                q = 0.0
            action[base + idx] = q
        return action


class HysteresisCapBankExpert(CapBankExpert):
    """Expert that changes action only when voltages exit a wider band.

    This mimics human operators who avoid frequent switching by using
    separate thresholds for turning capacitors on and off.
    """

    def __init__(self, env: IEEE33Env, v_on: float = 0.985, v_off: float = 1.015):
        super().__init__(env)
        self.v_on = v_on
        self.v_off = v_off
        self._current = np.zeros(env.action_space.shape[0])

    def act(self, env: IEEE33Env):
        action = CapBankExpert.act(self, env)
        self._current = action.copy()
        sim = env.unwrapped.simulator
        base = 0
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = np.abs(sim.buses[dev.bus_id].v)
            if bus_v < self.v_on:
                q = dev.q_max * sim.baseMVA
            elif bus_v > self.v_off:
                q = dev.q_min * sim.baseMVA
            else:
                q = self._current[idx]
            action[base + idx] = q
        self._current[: len(self.cap_ids)] = action[: len(self.cap_ids)]
        return action


class OLTCExpert(CapBankExpert):
    """Heuristic agent for on-load tap changers."""

    def act(self, env: IEEE33Env):
        action = CapBankExpert.act(self, env)
        sim = env.unwrapped.simulator
        gen_ids = [i for i, d in sim.devices.items() if isinstance(d, Generator) and not d.is_slack]
        des_ids = [i for i, d in sim.devices.items() if isinstance(d, StorageUnit)]
        base = 2 * len(gen_ids) + 2 * len(des_ids) + len(self.cap_ids)
        for idx, dev_id in enumerate(self.oltc_ids):
            dev = sim.devices[dev_id]
            v = np.abs(sim.buses[dev.t_bus].v)
            if v < self.v_min:
                tap = dev.tap_max
            elif v > self.v_max:
                tap = dev.tap_min
            else:
                tap = dev.tap
            action[base + idx] = tap
        return action


class RenewableGenExpert(CapBankExpert):
    """Heuristic agent controlling renewable generators."""

    def act(self, env: IEEE33Env):
        action = CapBankExpert.act(self, env)
        sim = env.unwrapped.simulator
        gen_ids = [i for i, d in sim.devices.items() if isinstance(d, Generator) and not d.is_slack]
        base = 0
        for idx, dev_id in enumerate(gen_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = np.abs(sim.buses[gen.bus_id].v)
                if v > self.v_max:
                    p = max(gen.p_min, 0.9 * gen.p_pot)
                else:
                    p = gen.p_pot
                action[base + idx] = p * sim.baseMVA
        return action
