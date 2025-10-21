"""Create algorithmically diverse controllers with renewable Q control."""

import numpy as np
from scipy.optimize import minimize
from collections import deque
from ready_to_use_l5_implementation import IEEE33ProperEnvironment


class L0_Random:
    """L0: Pure random control - worst performance."""
    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()

    def act(self, env):
        # Create truly random action matching action space exactly
        action = np.zeros(17)

        # Renewables P (0-4): from action space bounds
        action[0:3] = self._rng.uniform(0, 0.05, size=3)
        action[3:5] = self._rng.uniform(0, 0.1, size=2)

        # Renewables Q (5-9): reactive power control
        action[5:8] = self._rng.uniform(-0.02, 0.02, size=3)
        action[8:10] = self._rng.uniform(-0.04, 0.04, size=2)

        # Capacitors (10-15) - reduced to 1/8th
        action[10:12] = self._rng.uniform(-0.125, 0.125, size=2)
        action[12] = self._rng.uniform(0, 0.01875)
        action[13] = self._rng.uniform(0, 0.0125)
        action[14] = self._rng.uniform(0, 0.025)
        action[15] = self._rng.uniform(0, 0.01875)

        # OLTC (16): [0.9, 1.1]
        action[16] = self._rng.uniform(0.9, 1.1)

        assert env.action_space.contains(action), f"L0_Random generated invalid action"
        return action


class L1_BangBang:
    """L1: Bang-bang control with reactive power support."""
    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()
        self.v_threshold = 0.96

    def act(self, env):
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = voltages.min()

        action = np.zeros(17)

        if v_min < self.v_threshold:
            # FULL POWER - use both P and Q
            action[0:3] = 0.05
            action[3:5] = 0.10
            # Inject reactive power to boost voltage
            action[5:8] = 0.015   # 75% of max Q for solar
            action[8:10] = 0.03   # 75% of max Q for wind
            # All capacitors ON (reduced to 1/8th)
            action[10:16] = [0.0125, 0.0125, 0.001875, 0.00125, 0.0025, 0.001875]
            # OLTC boost
            action[16] = 0.95
        else:
            # MINIMUM POWER
            action[0:3] = 0.01
            action[3:5] = 0.02
            # Small reactive support
            action[5:8] = 0.005
            action[8:10] = 0.01
            # No capacitors
            action[10:16] = 0
            # OLTC neutral
            action[16] = 1.0

        return action


class L2_Proportional:
    """L2: Proportional control with reactive power."""
    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()
        self.kp_renewable = 10.0
        self.kp_caps = 50.0
        self.kp_q = 0.5  # Proportional gain for reactive power
        self.v_ref = 0.98

    def act(self, env):
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = voltages.min()
        v_avg = voltages.mean()

        action = np.zeros(17)

        # Proportional control
        error = self.v_ref - v_min

        # Renewable P: proportional to average voltage error
        avg_error = 1.0 - v_avg
        renewable_factor = np.clip(self.kp_renewable * avg_error, 0, 1)
        action[0:3] = 0.05 * renewable_factor
        action[3:5] = 0.10 * renewable_factor

        # Renewable Q: proportional to voltage error
        q_factor = np.clip(self.kp_q * error, 0, 1)
        action[5:8] = 0.02 * q_factor   # Use Q to support voltage
        action[8:10] = 0.04 * q_factor

        # Capacitors: proportional to min voltage error
        if error > 0:
            n_caps = int(np.clip(self.kp_caps * error, 0, 6))
            if n_caps >= 1: action[10] = 0.1
            if n_caps >= 2: action[11] = 0.1
            if n_caps >= 3: action[12] = 0.015
            if n_caps >= 4: action[13] = 0.01
            if n_caps >= 5: action[14] = 0.02
            if n_caps >= 6: action[15] = 0.015

        # OLTC: small adjustments
        action[16] = 1.0 - np.clip(error * 0.5, -0.05, 0.05)

        return action


class L3_PI_Controller:
    """L3: Proportional-Integral control with reactive power (FIXED)."""
    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()
        self.kp = 5.0  # Reduced from 8.0
        self.ki = 0.5  # Reduced from 2.0
        self.kp_q = 1.0  # Increased from 0.3 for better Q usage
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = 0.2  # Reduced from 0.5
        self.v_ref = 0.97  # FIXED: was 0.985, now proper target

    def act(self, env):
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = voltages.min()

        action = np.zeros(17)

        # PI control with anti-windup
        error = self.v_ref - v_min

        # Only accumulate integral when error is significant and not saturated
        if abs(error) > 0.001:
            self.integral += error
            self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)

        control_signal = self.kp * error + self.ki * self.integral

        # Renewable P (conservative)
        renewable_p = np.clip(control_signal * 0.08, 0, 0.5)
        action[0:3] = 0.03 * renewable_p
        action[3:5] = 0.06 * renewable_p

        # Renewable Q - more aggressive Q usage
        q_signal = self.kp_q * (error + self.integral)
        q_factor = np.clip(q_signal, 0, 1)
        action[5:8] = 0.015 * q_factor  # Increased from 0.02 max
        action[8:10] = 0.030 * q_factor  # Increased from 0.04 max

        # Capacitors based on control signal (reduced to 1/8th)
        if control_signal > 0.3:  # Lower threshold
            action[10:12] = [0.0125, 0.0125]
        if control_signal > 0.6:
            action[12:14] = [0.001875, 0.00125]
        if control_signal > 1.0:
            action[14:16] = [0.0025, 0.001875]

        # OLTC adjustment - FIXED direction
        oltc_adjustment = np.clip(error * 1.5, -0.05, 0.05)
        action[16] = 1.0 + oltc_adjustment  # FIXED: was 1.0 - oltc_adjustment

        self.prev_error = error

        return action


class L4_RuleBasedExpert:
    """L4: Rule-based expert with reactive power coordination."""
    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()
        self.voltage_history = deque(maxlen=10)

    def act(self, env):
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = voltages.min()
        v_max = voltages.max()
        v_avg = voltages.mean()
        v_std = voltages.std()

        self.voltage_history.append(v_min)

        action = np.zeros(17)

        # Rule 1: Emergency response
        if v_min < 0.94:
            action[0:5] = [0.05, 0.05, 0.05, 0.10, 0.10]
            action[5:10] = [0.018, 0.018, 0.018, 0.036, 0.036]  # Max Q support
            action[10:14] = [0.0125, 0.0125, 0.001875, 0.00125]  # Caps reduced to 1/8th
            action[16] = 0.95

        # Rule 2: Voltage trend analysis
        elif len(self.voltage_history) >= 3:
            trend = np.mean(list(self.voltage_history)[-3:]) - np.mean(list(self.voltage_history)[:3])

            if trend < -0.005:  # Voltage dropping - increase Q
                action[0:3] = 0.04
                action[3:5] = 0.08
                action[5:8] = 0.012   # 60% Q
                action[8:10] = 0.024
                action[10:12] = [0.0125, 0.0125]  # Caps reduced to 1/8th
                action[16] = 0.98

            elif trend > 0.005:  # Voltage rising - reduce Q
                action[0:3] = 0.02
                action[3:5] = 0.04
                action[5:8] = 0.005   # 25% Q
                action[8:10] = 0.01
                action[10] = 0.0125 if v_min < 0.97 else 0  # Caps reduced to 1/8th
                action[16] = 1.0

            else:  # Stable - moderate Q support
                if v_min < 0.97:
                    action[0:3] = 0.03
                    action[3:5] = 0.06
                    action[5:8] = 0.008   # 40% Q
                    action[8:10] = 0.016
                    action[10:12] = [0.0125, 0.0125]  # Caps reduced to 1/8th
                else:
                    action[0:3] = 0.025
                    action[3:5] = 0.05
                    action[5:8] = 0.006   # 30% Q
                    action[8:10] = 0.012
                    action[10] = 0.0125  # Caps reduced to 1/8th
                action[16] = 1.0

        # Rule 3: High voltage variance - distributed Q support
        elif v_std > 0.01:
            action[0:5] = [0.03, 0.03, 0.03, 0.06, 0.06]
            action[5:10] = [0.01, 0.01, 0.01, 0.02, 0.02]  # Distributed Q
            action[10:13] = [0.0125, 0.0125, 0.001875]  # Caps reduced to 1/8th
            action[16] = 0.99

        # Rule 4: Normal operation
        else:
            error = 0.98 - v_min
            if error > 0.02:
                action[0:3] = 0.04
                action[3:5] = 0.08
                action[5:8] = 0.01
                action[8:10] = 0.02
                action[10:12] = [0.1, 0.1]
            elif error > 0.01:
                action[0:3] = 0.03
                action[3:5] = 0.06
                action[5:8] = 0.007
                action[8:10] = 0.014
                action[10] = 0.1
            else:
                action[0:3] = 0.02
                action[3:5] = 0.04
                action[5:8] = 0.004
                action[8:10] = 0.008

            action[16] = np.clip(1.0 - error * 2, 0.95, 1.05)

        return action


class L5_ScipyOptimal:
    """L5: Scipy optimization that directly optimizes environment reward via simulation."""

    def __init__(self, env):
        self.env = env
        self._rng = np.random.default_rng()
        self.cap_ratings = np.array([1.0, 1.0, 0.15, 0.1, 0.2, 0.15])
        self.renewable_p_max = np.array([0.05, 0.05, 0.05, 0.10, 0.10])
        self.renewable_q_max = np.array([0.02, 0.02, 0.02, 0.04, 0.04])

        self.curtailment_weight = getattr(env, 'curtailment_weight', 1.0)
        self.capacitor_switch_cost = getattr(env, 'capacitor_switch_cost', 1.0)
        self.oltc_switch_cost = getattr(env, 'oltc_switch_cost', 5.0)
        self.baseMVA = env.unwrapped.simulator.baseMVA
        self.delta_t = env.unwrapped.simulator.delta_t

        self.prev_action = None
        self.steps_since_switch = 0
        self.min_steps_between_switches = 5

        # Cache for simulator state during optimization
        self._saved_state = None
        self._eval_count = 0

    def objective(self, x, simulator_state, p_pot_values, p_load_dict):
        """Objective: get actual reward by running power flow on saved state."""
        import copy

        self._eval_count += 1

        # Build action from x
        action = np.zeros(17)
        action[0:5] = x[:5]      # Renewable P
        action[5:10] = x[5:10]   # Renewable Q

        # Binary capacitors
        for i in range(6):
            if x[10+i] > 0.5:
                action[10+i] = self.cap_ratings[i] / 10.0

        action[16] = x[16]  # OLTC

        # Evaluate by running simulator transition
        try:
            sim = copy.deepcopy(simulator_state)

            # Convert action to transition() parameters (like ANMEnv.step() does)
            from gym_anm.simulator.components import RenewableGen, CapacitorBank, OLTC

            # Get device IDs by type
            gen_ids = [dev_id for dev_id, dev in sim.devices.items() if isinstance(dev, RenewableGen)]
            cap_ids = [dev_id for dev_id, dev in sim.devices.items() if isinstance(dev, CapacitorBank)]
            oltc_ids = [dev_id for dev_id, dev in sim.devices.items() if isinstance(dev, OLTC)]

            # Build P_pot dictionary
            # CRITICAL: Must match what environment does!
            # The environment calls next_vars() which returns P_pot values.
            # We receive p_pot_values from act() which are from next_vars()
            P_pot_dict = {}
            for i, dev_id in enumerate(gen_ids):
                P_pot_dict[dev_id] = p_pot_values[i]

            # Build P_set_points dictionary (renewable P)
            P_set_points = {}
            for i, dev_id in enumerate(gen_ids):
                P_set_points[dev_id] = action[i]

            # Build Q_set_points dictionary (renewable Q + capacitors)
            Q_set_points = {}
            for i, dev_id in enumerate(gen_ids):
                Q_set_points[dev_id] = action[5 + i]
            for i, dev_id in enumerate(cap_ids):
                Q_set_points[dev_id] = action[10 + i]

            # Build tap_set_points dictionary (OLTC)
            tap_set_points = {}
            for i, dev_id in enumerate(oltc_ids):
                tap_set_points[dev_id] = action[16 + i]

            # Run transition with dictionaries
            _, r, e_loss, penalty, pfe_converged = sim.transition(
                p_load_dict,
                P_pot_dict,
                P_set_points,
                Q_set_points,
                tap_set_points,
            )

            # Check if power flow converged
            if not pfe_converged:
                return 1000.0

            # Environment IGNORES r and recomputes from e_loss and penalty!
            # Need to match env's reward computation exactly
            from gym_anm.simulator.components import RenewableGen

            # Clip e_loss and penalty like the environment does
            clipped_e_loss = np.sign(e_loss) * np.clip(np.abs(e_loss), 0, 100.0)
            clipped_penalty = np.clip(penalty, 0, 1000.0)

            # Base reward from environment
            base_reward = -(clipped_e_loss + clipped_penalty)

            # Add ADDITIONAL curtailment penalty (40 - 1)x
            curtailment = 0.0
            for dev_id, dev in sim.devices.items():
                if isinstance(dev, RenewableGen):
                    curtailment += max(0, dev.p_pot - dev.p)

            # Apply ADDITIONAL curtailment weight (40 - 1 = 39x more)
            additional_curtailment_penalty = curtailment * (40.0 - 1.0) * sim.delta_t

            # Total reward = base_reward - additional_penalty
            total_reward = base_reward - additional_curtailment_penalty

            # Return cost to minimize (negate reward)
            return -total_reward

        except Exception as e:
            import traceback
            print(f"ERROR in objective: {e}")
            traceback.print_exc()
            return 1000.0

    def act(self, env):
        import copy

        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = voltages.min()
        v_max = voltages.max()

        # Emergency response - ONLY for extreme violations
        # Removed: was triggering always and preventing scipy optimization

        # Removed hysteresis - L5 should optimize every step to be truly optimal

        # Initialize
        x0 = np.zeros(17)

        if v_min < 0.95:
            x0[0:5] = self.renewable_p_max * 0.8
            x0[5:10] = self.renewable_q_max * 0.6
            x0[10:16] = [1, 1, 1, 1, 0, 0]
            x0[16] = 0.97
        elif v_min < 0.96:
            x0[0:5] = self.renewable_p_max * 0.5
            x0[5:10] = self.renewable_q_max * 0.4
            x0[10:16] = [1, 1, 0, 0, 0, 0]
            x0[16] = 0.99
        else:
            x0[0:5] = self.renewable_p_max * 0.3
            x0[5:10] = self.renewable_q_max * 0.2
            x0[10:16] = [0, 0, 0, 0, 0, 0]
            x0[16] = 1.0

        # Bounds
        bounds = []
        for i in range(5):
            bounds.append((0, self.renewable_p_max[i]))
        for i in range(5):
            bounds.append((-self.renewable_q_max[i], self.renewable_q_max[i]))
        for i in range(6):
            bounds.append((0, 1))
        bounds.append((0.95, 1.05))

        # Optimize
        self._eval_count = 0
        try:
            # Save simulator state for objective evaluation
            simulator_state = copy.deepcopy(sim)

            # Get load and renewable values FROM next_vars() like the environment does!
            # This is CRITICAL - we must match exactly what the environment will pass to transition()
            from gym_anm.simulator.components import RenewableGen, Load, Generator

            # Call next_vars() to get the values that will be used
            vars = env.next_vars(env.state)
            P_load = vars[: env.simulator.N_load]
            P_pot = vars[env.simulator.N_load : env.simulator.N_load + env.simulator.N_non_slack_gen]

            # Build dictionaries like ANMEnv.step() does
            load_idx, gen_idx = 0, 0
            p_load_dict, p_pot_dict_from_next_vars = {}, {}
            for dev_id, dev in sim.devices.items():
                if isinstance(dev, Load):
                    p_load_dict[dev_id] = P_load[load_idx]
                    load_idx += 1
                elif isinstance(dev, Generator) and not dev.is_slack:
                    p_pot_dict_from_next_vars[dev_id] = P_pot[gen_idx]
                    gen_idx += 1

            # For passing to objective - use the P_pot values from next_vars()
            p_pot_values = [p_pot_dict_from_next_vars[dev_id] for dev_id in sorted(p_pot_dict_from_next_vars.keys())]

            result = minimize(
                lambda x: self.objective(x, simulator_state, p_pot_values, p_load_dict),
                x0,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 5, 'ftol': 1e-3}  # Fast optimization
            )

            if result.success:
                action = np.zeros(17)
                action[0:5] = result.x[:5]
                action[5:10] = result.x[5:10]

                # Binary capacitors
                for i in range(6):
                    if result.x[10+i] > 0.5:
                        action[10+i] = self.cap_ratings[i] / 10.0

                action[16] = result.x[16]

                self.prev_action = action.copy()
                self.steps_since_switch = 0
                return action

        except Exception as e:
            print(f"Optimization failed: {e}")

        # Fallback
        action = np.zeros(17)
        action[0:5] = x0[0:5]
        action[5:10] = x0[5:10]
        for i in range(6):
            if x0[10+i] > 0.5:
                action[10+i] = self.cap_ratings[i] / 10.0
        action[16] = x0[16]

        self.prev_action = action.copy()
        return action


def test_algorithmic_diversity():
    """Test controllers with Q control."""
    print("="*80)
    print("TESTING CONTROLLERS WITH RENEWABLE REACTIVE POWER")
    print("="*80)

    test_scenarios = [
        ("High Load (1.5x)", 1.5),
        ("Very High Load (1.8x)", 1.8),
    ]

    all_results = {}

    for scenario_name, load_pattern in test_scenarios:
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"{'='*80}")

        results = {}

        for name, controller_class in [
            ('L0', L0_Random),
            ('L1', L1_BangBang),
            ('L2', L2_Proportional),
            ('L3', L3_PI_Controller),
            ('L4', L4_RuleBasedExpert),
            ('L5', L5_ScipyOptimal),
        ]:
            env = IEEE33ProperEnvironment(load_scale=load_pattern)
            controller = controller_class(env)

            obs, _ = env.reset(seed=42)
            total_reward = 0
            violations = 0
            q_usage = []

            for step in range(96):
                action = controller.act(env)
                q_usage.append(action[5:10].copy())
                obs, reward, done, _, info = env.step(action)
                total_reward += reward
                if info.get('dev_penalty', 0) > 0:
                    violations += 1
                if done:
                    break

            q_usage = np.array(q_usage)
            results[name] = {
                'reward': total_reward,
                'violations': violations,
                'q_mean': q_usage.mean(),
                'q_std': q_usage.std(),
            }

            print(f"\n{name}:")
            print(f"  Reward: {total_reward:.2f}")
            print(f"  Violations: {violations}/96")
            print(f"  Q usage - mean: {q_usage.mean():.4f}, std: {q_usage.std():.4f}")

        all_results[scenario_name] = results

    return all_results


if __name__ == "__main__":
    results = test_algorithmic_diversity()
