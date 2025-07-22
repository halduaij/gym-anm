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


class BaseHeuristic:
    """Base class for all heuristic policies with voltage thresholds."""

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
        self.gen_non_slack_ids = [
            i for i, d in sim.devices.items() if isinstance(d, Generator) and not d.is_slack
        ]
        self.des_ids = [i for i, d in sim.devices.items() if isinstance(d, StorageUnit)]

    def get_base_action(self, env: IEEE33Env):
        """Get a base action with default values for all devices."""
        sim = env.unwrapped.simulator
        gen_ids = self.gen_non_slack_ids
        N_gen = len(gen_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        N_oltc = len(self.oltc_ids)
        
        action = np.zeros(env.action_space.shape[0])
        
        base = 0
        # Default: all generators at their potential
        for idx, dev_id in enumerate(gen_ids):
            gen = sim.devices[dev_id]
            action[base + idx] = gen.p_pot * sim.baseMVA
        base += N_gen
        
        # Default: no reactive power from generators
        base += N_gen
        
        # Default: no storage activity
        base += 2 * N_des
        
        # Default: no capacitor banks
        base += N_cap
        
        # Default: all OLTCs at nominal tap
        for idx, dev_id in enumerate(self.oltc_ids):
            action[base + idx] = sim.devices[dev_id].tap
        
        return action


# Capacitor Bank Heuristics
class CapBankHeuristic(BaseHeuristic):
    """General capacitor bank heuristic with configurable thresholds."""

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        base = 2 * N_gen + 2 * N_des
        
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


class SimpleCapBankHeuristic(CapBankHeuristic):
    """Heuristic with 0.99/1.01 voltage thresholds."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.99, v_max=1.01)


class ConservativeCapBankHeuristic(CapBankHeuristic):
    """Heuristic that acts only for larger voltage deviations."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.98, v_max=1.02)


class AggressiveCapBankHeuristic(CapBankHeuristic):
    """Heuristic that reacts to small voltage deviations."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.995, v_max=1.005)


class NoisyCapBankHeuristic(CapBankHeuristic):
    """Heuristic that senses voltages with Gaussian noise."""

    def __init__(self, env: IEEE33Env, noise_std: float = 0.005):
        super().__init__(env)
        self.noise_std = noise_std

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        base = 2 * N_gen + 2 * N_des
        
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


class DelayedCapBankHeuristic(CapBankHeuristic):
    """Heuristic that only updates its action every ``delay`` steps."""

    def __init__(self, env: IEEE33Env, delay: int = 2):
        super().__init__(env)
        self.delay = max(1, delay)
        self._counter = 0
        self._last_action = None

    def act(self, env: IEEE33Env):
        if self._counter % self.delay == 0:
            self._last_action = super().act(env)
        self._counter += 1
        if self._last_action is None:
            return self.get_base_action(env)
        return self._last_action


class LaggingCapBankHeuristic(CapBankHeuristic):
    """Heuristic using voltage measurements from ``lag`` steps ago."""

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
            self._history = self._history[-self.lag - 1:]
        
        action = self.get_base_action(env)
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        base = 2 * N_gen + 2 * N_des
        
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


class HysteresisCapBankHeuristic(CapBankHeuristic):
    """Heuristic that changes action only when voltages exit a wider band."""

    def __init__(self, env: IEEE33Env, v_on: float = 0.985, v_off: float = 1.015):
        super().__init__(env)
        self.v_on = v_on
        self.v_off = v_off
        self._current_q = {}

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        base = 2 * N_gen + 2 * N_des
        
        for idx, dev_id in enumerate(self.cap_ids):
            dev = sim.devices[dev_id]
            bus_v = np.abs(sim.buses[dev.bus_id].v)
            
            current_q = self._current_q.get(dev_id, 0.0)
            
            if bus_v < self.v_on:
                q = dev.q_max * sim.baseMVA
            elif bus_v > self.v_off:
                q = dev.q_min * sim.baseMVA
            else:
                q = current_q
            
            action[base + idx] = q
            self._current_q[dev_id] = q
        
        return action


# OLTC Heuristics
class OLTCHeuristic(BaseHeuristic):
    """Basic OLTC heuristic with voltage control."""

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        base = 2 * N_gen + 2 * N_des + N_cap
        
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


class SimpleOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic with standard voltage thresholds."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.99, v_max=1.01)


class ConservativeOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic that acts only for larger deviations."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.98, v_max=1.02)


class AggressiveOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic that reacts to small deviations."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.995, v_max=1.005)


class NoisyOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic with noisy voltage measurements."""

    def __init__(self, env: IEEE33Env, noise_std: float = 0.005):
        super().__init__(env)
        self.noise_std = noise_std

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        base = 2 * N_gen + 2 * N_des + N_cap
        
        for idx, dev_id in enumerate(self.oltc_ids):
            dev = sim.devices[dev_id]
            v = np.abs(sim.buses[dev.t_bus].v)
            v += np.random.normal(0.0, self.noise_std)
            if v < self.v_min:
                tap = dev.tap_max
            elif v > self.v_max:
                tap = dev.tap_min
            else:
                tap = dev.tap
            action[base + idx] = tap
        return action


class DelayedOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic that delays tap changes."""

    def __init__(self, env: IEEE33Env, delay: int = 5):
        super().__init__(env)
        self.delay = max(1, delay)
        self._counter = 0
        self._last_taps = {}

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        base = 2 * N_gen + 2 * N_des + N_cap
        
        if self._counter % self.delay == 0:
            # Update tap positions
            for idx, dev_id in enumerate(self.oltc_ids):
                dev = sim.devices[dev_id]
                v = np.abs(sim.buses[dev.t_bus].v)
                if v < self.v_min:
                    tap = dev.tap_max
                elif v > self.v_max:
                    tap = dev.tap_min
                else:
                    tap = dev.tap
                self._last_taps[dev_id] = tap
        
        # Use last computed taps
        for idx, dev_id in enumerate(self.oltc_ids):
            action[base + idx] = self._last_taps.get(dev_id, sim.devices[dev_id].tap)
        
        self._counter += 1
        return action


class HysteresisOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic with hysteresis bands."""

    def __init__(self, env: IEEE33Env, v_low: float = 0.985, v_high: float = 1.015):
        super().__init__(env)
        self.v_low = v_low
        self.v_high = v_high
        self._current_taps = {}

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        base = 2 * N_gen + 2 * N_des + N_cap
        
        for idx, dev_id in enumerate(self.oltc_ids):
            dev = sim.devices[dev_id]
            v = np.abs(sim.buses[dev.t_bus].v)
            
            current_tap = self._current_taps.get(dev_id, dev.tap)
            
            if v < self.v_low:
                tap = dev.tap_max
            elif v > self.v_high:
                tap = dev.tap_min
            else:
                tap = current_tap
            
            action[base + idx] = tap
            self._current_taps[dev_id] = tap
        
        return action


class DeadbandOLTCHeuristic(OLTCHeuristic):
    """OLTC heuristic with deadband to prevent hunting."""

    def __init__(self, env: IEEE33Env, deadband: float = 0.005):
        super().__init__(env)
        self.deadband = deadband

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        base = 2 * N_gen + 2 * N_des + N_cap
        
        for idx, dev_id in enumerate(self.oltc_ids):
            dev = sim.devices[dev_id]
            v = np.abs(sim.buses[dev.t_bus].v)
            
            # Only change tap if voltage is outside deadband
            if v < self.v_min - self.deadband:
                tap = dev.tap_max
            elif v > self.v_max + self.deadband:
                tap = dev.tap_min
            else:
                tap = dev.tap
            
            action[base + idx] = tap
        return action


# Renewable Generation Heuristics
class RenewableGenHeuristic(BaseHeuristic):
    """Basic renewable generation heuristic with voltage-based curtailment."""

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        base = 0
        
        for idx, dev_id in enumerate(self.gen_non_slack_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = np.abs(sim.buses[gen.bus_id].v)
                if v > self.v_max:
                    p = max(gen.p_min, 0.9 * gen.p_pot)
                else:
                    p = gen.p_pot
                action[base + idx] = p * sim.baseMVA
        return action


class SimpleRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic with standard thresholds."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.99, v_max=1.01)


class ConservativeRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic that curtails only for large overvoltages."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.98, v_max=1.02)


class AggressiveRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic that curtails aggressively."""
    def __init__(self, env: IEEE33Env):
        super().__init__(env, v_min=0.995, v_max=1.005)


class ProportionalRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic with proportional curtailment."""

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        base = 0
        
        for idx, dev_id in enumerate(self.gen_non_slack_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = np.abs(sim.buses[gen.bus_id].v)
                if v > self.v_max:
                    # Proportional curtailment based on overvoltage
                    curtailment = min(1.0, (v - self.v_max) / 0.02)
                    p = gen.p_pot * (1 - 0.5 * curtailment)
                    p = max(gen.p_min, p)
                else:
                    p = gen.p_pot
                action[base + idx] = p * sim.baseMVA
        return action


class SteppedRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic with stepped curtailment levels."""

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        base = 0
        
        for idx, dev_id in enumerate(self.gen_non_slack_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = np.abs(sim.buses[gen.bus_id].v)
                if v > 1.02:
                    p = gen.p_pot * 0.5
                elif v > 1.015:
                    p = gen.p_pot * 0.7
                elif v > 1.01:
                    p = gen.p_pot * 0.9
                else:
                    p = gen.p_pot
                p = max(gen.p_min, p)
                action[base + idx] = p * sim.baseMVA
        return action


class NoisyRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic with noisy voltage measurements."""

    def __init__(self, env: IEEE33Env, noise_std: float = 0.005):
        super().__init__(env)
        self.noise_std = noise_std

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        base = 0
        
        for idx, dev_id in enumerate(self.gen_non_slack_ids):
            gen = sim.devices[dev_id]
            if dev_id in self.ren_gen_ids:
                v = np.abs(sim.buses[gen.bus_id].v)
                v += np.random.normal(0.0, self.noise_std)
                if v > self.v_max:
                    p = max(gen.p_min, 0.9 * gen.p_pot)
                else:
                    p = gen.p_pot
                action[base + idx] = p * sim.baseMVA
        return action


class DelayedRenewableHeuristic(RenewableGenHeuristic):
    """Renewable heuristic with delayed response."""

    def __init__(self, env: IEEE33Env, delay: int = 3):
        super().__init__(env)
        self.delay = max(1, delay)
        self._counter = 0
        self._last_power = {}

    def act(self, env: IEEE33Env):
        action = self.get_base_action(env)
        sim = env.unwrapped.simulator
        base = 0
        
        if self._counter % self.delay == 0:
            # Update power setpoints
            for idx, dev_id in enumerate(self.gen_non_slack_ids):
                gen = sim.devices[dev_id]
                if dev_id in self.ren_gen_ids:
                    v = np.abs(sim.buses[gen.bus_id].v)
                    if v > self.v_max:
                        p = max(gen.p_min, 0.9 * gen.p_pot)
                    else:
                        p = gen.p_pot
                    self._last_power[dev_id] = p * sim.baseMVA
        
        # Use last computed power
        for idx, dev_id in enumerate(self.gen_non_slack_ids):
            if dev_id in self.ren_gen_ids and dev_id in self._last_power:
                action[base + idx] = self._last_power[dev_id]
        
        self._counter += 1
        return action


# Combined Heuristics
class CombinedHeuristic(BaseHeuristic):
    """Combines capacitor, OLTC, and renewable control."""

    def __init__(self, env: IEEE33Env):
        super().__init__(env)
        self.cap_heuristic = CapBankHeuristic(env)
        self.oltc_heuristic = OLTCHeuristic(env)
        self.ren_heuristic = RenewableGenHeuristic(env)

    def act(self, env: IEEE33Env):
        # Start with base action
        action = self.get_base_action(env)
        
        # Apply each heuristic's logic
        cap_action = self.cap_heuristic.act(env)
        oltc_action = self.oltc_heuristic.act(env)
        ren_action = self.ren_heuristic.act(env)
        
        # Combine the actions
        sim = env.unwrapped.simulator
        N_gen = len(self.gen_non_slack_ids)
        N_des = len(self.des_ids)
        N_cap = len(self.cap_ids)
        
        # Use renewable generation control
        action[:N_gen] = ren_action[:N_gen]
        
        # Use capacitor control
        base_cap = 2 * N_gen + 2 * N_des
        action[base_cap:base_cap + N_cap] = cap_action[base_cap:base_cap + N_cap]
        
        # Use OLTC control
        base_oltc = base_cap + N_cap
        action[base_oltc:] = oltc_action[base_oltc:]
        
        return action


class RandomHeuristic(BaseHeuristic):
    """Completely random actions for baseline comparison."""

    def act(self, env: IEEE33Env):
        return env.action_space.sample()


class DoNothingHeuristic(BaseHeuristic):
    """Does nothing - keeps all devices at default/nominal values."""

    def act(self, env: IEEE33Env):
        return self.get_base_action(env)


# Optimization-based Expert Policies
class OptimizationBasedExpert(BaseHeuristic):
    """
    A more sophisticated expert using optimization techniques.
    This is closer to a true expert policy.
    """
    
    def __init__(self, env: IEEE33Env, horizon: int = 1):
        super().__init__(env)
        self.horizon = horizon
        try:
            import cvxpy as cp
            self.cp = cp
        except ImportError:
            raise ImportError("cvxpy required for OptimizationBasedExpert. Install with: pip install cvxpy")
    
    def act(self, env: IEEE33Env):
        """
        Solve a simplified optimal power flow problem.
        Uses DC power flow approximation for tractability.
        """
        sim = env.unwrapped.simulator
        cp = self.cp
        
        # Get current state
        v_mag = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        
        # Decision variables
        N_gen = len(self.gen_non_slack_ids)
        N_cap = len(self.cap_ids)
        N_oltc = len(self.oltc_ids)
        
        # Create optimization variables
        p_gen = cp.Variable(N_gen)
        q_cap = cp.Variable(N_cap)
        tap_oltc = cp.Variable(N_oltc)
        
        # Approximate voltage magnitudes (linearized around current operating point)
        # This is a simplified sensitivity-based approach
        v_sens_p = 0.1  # Voltage sensitivity to active power (simplified)
        v_sens_q = 0.2  # Voltage sensitivity to reactive power (simplified)
        v_sens_tap = 0.05  # Voltage sensitivity to tap changes (simplified)
        
        # Objective: Minimize losses (approximated by generation)
        objective = cp.Minimize(cp.sum(p_gen))
        
        # Constraints
        constraints = []
        
        # Generator limits
        for i, dev_id in enumerate(self.gen_non_slack_ids):
            gen = sim.devices[dev_id]
            constraints.append(p_gen[i] >= gen.p_min * sim.baseMVA)
            constraints.append(p_gen[i] <= gen.p_pot * sim.baseMVA)
        
        # Capacitor limits
        for i, dev_id in enumerate(self.cap_ids):
            cap = sim.devices[dev_id]
            constraints.append(q_cap[i] >= cap.q_min * sim.baseMVA)
            constraints.append(q_cap[i] <= cap.q_max * sim.baseMVA)
        
        # OLTC limits
        for i, dev_id in enumerate(self.oltc_ids):
            oltc = sim.devices[dev_id]
            constraints.append(tap_oltc[i] >= oltc.tap_min)
            constraints.append(tap_oltc[i] <= oltc.tap_max)
        
        # Voltage constraints (simplified)
        for i, bus in enumerate(sim.buses.values()):
            if bus.is_slack:
                continue
            
            # Approximate voltage change
            dv = 0
            
            # Contribution from generators (if connected to this bus)
            for j, gen_id in enumerate(self.gen_non_slack_ids):
                if sim.devices[gen_id].bus_id == bus.bus_id:
                    dv += v_sens_p * (p_gen[j] - sim.devices[gen_id].p * sim.baseMVA)
            
            # Contribution from capacitors (if connected to this bus)
            for j, cap_id in enumerate(self.cap_ids):
                if sim.devices[cap_id].bus_id == bus.bus_id:
                    dv += v_sens_q * q_cap[j]
            
            # Voltage limits
            v_current = v_mag[i]
            constraints.append(v_current + dv >= 0.95)
            constraints.append(v_current + dv <= 1.05)
        
        # Solve the optimization problem
        prob = cp.Problem(objective, constraints)
        
        try:
            prob.solve(solver=cp.ECOS)
            
            if prob.status in ["optimal", "optimal_inaccurate"]:
                # Extract solution
                action = self.get_base_action(env)
                
                # Set generator powers
                base = 0
                for i, dev_id in enumerate(self.gen_non_slack_ids):
                    action[base + i] = p_gen.value[i]
                
                # Set capacitor reactive powers
                base_cap = 2 * N_gen + 2 * len(self.des_ids)
                for i in range(N_cap):
                    action[base_cap + i] = q_cap.value[i]
                
                # Set OLTC taps
                base_oltc = base_cap + N_cap
                for i in range(N_oltc):
                    action[base_oltc + i] = tap_oltc.value[i]
                
                return action
            else:
                # Fallback to heuristic if optimization fails
                return CombinedHeuristic(env).act(env)
                
        except Exception as e:
            # Fallback to heuristic if optimization fails
            print(f"Optimization failed: {e}")
            return CombinedHeuristic(env).act(env)


class SensitivityBasedExpert(BaseHeuristic):
    """
    Expert using power flow sensitivities for near-optimal control.
    More practical than full optimization but better than simple heuristics.
    """
    
    def __init__(self, env: IEEE33Env):
        super().__init__(env)
        self._compute_sensitivities()
    
    def _compute_sensitivities(self):
        """
        Compute voltage sensitivities to control actions.
        In practice, these would be computed from the Jacobian matrix.
        """
        # Simplified sensitivity values (would be computed from power flow)
        self.dv_dp = 0.001  # Voltage change per MW of generation
        self.dv_dq = 0.002  # Voltage change per MVAr of reactive power
        self.dv_dtap = 0.05  # Voltage change per tap position
    
    def act(self, env: IEEE33Env):
        """
        Use sensitivities to make coordinated control decisions.
        """
        sim = env.unwrapped.simulator
        action = self.get_base_action(env)
        
        # Identify voltage violations
        voltage_errors = []
        for bus_id, bus in sim.buses.items():
            v = np.abs(bus.v)
            if v < self.v_min:
                voltage_errors.append((bus_id, v - self.v_min, 'low'))
            elif v > self.v_max:
                voltage_errors.append((bus_id, v - self.v_max, 'high'))
        
        # Sort by severity
        voltage_errors.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Address voltage violations using sensitivities
        for bus_id, error, direction in voltage_errors:
            
            # Use renewable curtailment for overvoltage
            if direction == 'high':
                base = 0
                for idx, dev_id in enumerate(self.gen_non_slack_ids):
                    if dev_id in self.ren_gen_ids:
                        gen = sim.devices[dev_id]
                        if gen.bus_id == bus_id or self._is_nearby(gen.bus_id, bus_id):
                            # Curtail generation proportionally to error
                            curtailment = min(0.5, abs(error) * 10)
                            p = gen.p_pot * (1 - curtailment)
                            action[base + idx] = max(gen.p_min, p) * sim.baseMVA
            
            # Use capacitors for undervoltage
            elif direction == 'low':
                N_gen = len(self.gen_non_slack_ids)
                N_des = len(self.des_ids)
                base_cap = 2 * N_gen + 2 * N_des
                
                for idx, dev_id in enumerate(self.cap_ids):
                    cap = sim.devices[dev_id]
                    if cap.bus_id == bus_id or self._is_nearby(cap.bus_id, bus_id):
                        # Switch on capacitor
                        action[base_cap + idx] = cap.q_max * sim.baseMVA
            
            # Use OLTC for both over and undervoltage
            N_gen = len(self.gen_non_slack_ids)
            N_des = len(self.des_ids)
            N_cap = len(self.cap_ids)
            base_oltc = 2 * N_gen + 2 * N_des + N_cap
            
            for idx, dev_id in enumerate(self.oltc_ids):
                oltc = sim.devices[dev_id]
                if oltc.t_bus == bus_id or self._is_nearby(oltc.t_bus, bus_id):
                    if direction == 'low':
                        # Increase tap to boost voltage
                        new_tap = min(oltc.tap_max, oltc.tap + 0.01)
                    else:
                        # Decrease tap to reduce voltage
                        new_tap = max(oltc.tap_min, oltc.tap - 0.01)
                    action[base_oltc + idx] = new_tap
        
        return action
    
    def _is_nearby(self, bus1: int, bus2: int, distance: int = 2) -> bool:
        """
        Check if two buses are electrically nearby.
        Simplified - in practice would use electrical distance.
        """
        return abs(bus1 - bus2) <= distance


class MPCBasedExpert(BaseHeuristic):
    """
    Model Predictive Control based expert.
    Uses a simplified model to predict future states and optimize control.
    """
    
    def __init__(self, env: IEEE33Env, horizon: int = 5):
        super().__init__(env)
        self.horizon = horizon
        self.past_actions = []
    
    def act(self, env: IEEE33Env):
        """
        MPC approach: optimize over a horizon but only apply first action.
        """
        sim = env.unwrapped.simulator
        
        # For demonstration, we use a rule-based approach that considers
        # future trends (in practice, this would solve an optimization problem)
        
        # Estimate load trend from history
        current_load = sum(abs(d.p) for d in sim.devices.values() if hasattr(d, 'qp_ratio'))
        
        # Get current worst voltage
        voltages = [np.abs(bus.v) for bus in sim.buses.values()]
        min_v, max_v = min(voltages), max(voltages)
        
        # Predictive control based on voltage trajectory
        action = self.get_base_action(env)
        
        # If voltage is trending toward limits, act preemptively
        if len(self.past_actions) > 0:
            # Simple trend estimation
            if max_v > 1.005 and max_v > self.last_max_v:
                # Voltage rising - curtail renewables more aggressively
                base = 0
                for idx, dev_id in enumerate(self.gen_non_slack_ids):
                    if dev_id in self.ren_gen_ids:
                        gen = sim.devices[dev_id]
                        # Preemptive curtailment
                        action[base + idx] = gen.p_pot * 0.8 * sim.baseMVA
            
            elif min_v < 0.995 and min_v < self.last_min_v:
                # Voltage dropping - prepare capacitors
                N_gen = len(self.gen_non_slack_ids)
                N_des = len(self.des_ids)
                base_cap = 2 * N_gen + 2 * N_des
                
                for idx, dev_id in enumerate(self.cap_ids):
                    cap = sim.devices[dev_id]
                    # Preemptive capacitor switching
                    action[base_cap + idx] = cap.q_max * sim.baseMVA
        
        # Store for trend analysis
        self.last_min_v = min_v
        self.last_max_v = max_v
        self.past_actions.append(action.copy())
        if len(self.past_actions) > self.horizon:
            self.past_actions.pop(0)
        
        return action


# Backward compatibility aliases
CapBankExpert = CapBankHeuristic
SimpleCapBankExpert = SimpleCapBankHeuristic
ConservativeCapBankExpert = ConservativeCapBankHeuristic
AggressiveCapBankExpert = AggressiveCapBankHeuristic
NoisyCapBankExpert = NoisyCapBankHeuristic
DelayedCapBankExpert = DelayedCapBankHeuristic
LaggingCapBankExpert = LaggingCapBankHeuristic
HysteresisCapBankExpert = HysteresisCapBankHeuristic
OLTCExpert = OLTCHeuristic
RenewableGenExpert = RenewableGenHeuristic