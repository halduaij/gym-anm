"""Algorithmic controller hierarchy for IEEE33 with renewable generators.

Hierarchy based on control complexity:
- L0: Random Control (baseline)
- L1: Bang-Bang Control (simple threshold switching)
- L2: Proportional Control (P-controller)
- L3: PI Control (Proportional-Integral)
- L4: MPC Control (Model Predictive Control)
- L5: Hierarchical MPC (Multi-timescale optimization)
"""

import numpy as np
from collections import deque


class L0_RandomControl:
    """Level 0: Random control actions (baseline)."""
    
    def __init__(self, env):
        self.env = env
        self.rng = np.random.RandomState(42)
        
    def act(self, env):
        """Generate random actions within bounds."""
        return env.action_space.sample()


class L1_BangBangControl:
    """Level 1: Bang-bang (on/off) control with fixed thresholds."""
    
    def __init__(self, env):
        self.env = env
        # Fixed thresholds
        self.v_low_thresh = 0.97
        self.v_high_thresh = 1.03
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get system voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Renewable control: Use all available renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot  # Use all available
        
        # No reactive from renewables (simple strategy)
        for i in range(5, 10):
            action[i] = 0.0
        
        # Bang-bang capacitor control
        if v_min < self.v_low_thresh:
            action[10] = 0.3  # Turn ON
            action[11] = 0.3
        elif v_max > self.v_high_thresh:
            action[10] = 0.0  # Turn OFF
            action[11] = 0.0
        else:
            action[10] = 0.0  # Default OFF
            action[11] = 0.0
        
        # Bang-bang OLTC control
        if v_min < self.v_low_thresh:
            action[12] = 0.95  # Boost voltage
        elif v_max > self.v_high_thresh:
            action[12] = 1.05  # Reduce voltage
        else:
            action[12] = 1.0  # Nominal
        
        return action


class L2_ProportionalControl:
    """Level 2: Proportional (P) control based on voltage error."""
    
    def __init__(self, env):
        self.env = env
        # Reduced proportional gains for stability
        self.kp_renewable = 5.0
        self.kp_reactive = 2.0
        self.kp_cap = 3.0
        self.kp_oltc = 5.0
        # Reference voltage
        self.v_ref = 1.0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get system voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Use worst-case voltage for control
        if abs(v_max - 1.05) > abs(0.95 - v_min):
            # Overvoltage is worse
            v_error = 1.0 - v_max
        else:
            # Undervoltage is worse
            v_error = 1.0 - v_min
        
        # Renewable P control - always use available renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                # Only curtail if severely overvoltage
                if v_max > 1.045:
                    curtailment = min(0.7, self.kp_renewable * (v_max - 1.045))
                    action[i] = gen.p_pot * (1 - curtailment)
                else:
                    action[i] = gen.p_pot
        
        # Proportional reactive power control
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            q_max = q_limits[i-5]
            # Inject Q if voltage low, absorb if high
            if v_min < 0.98:
                action[i] = q_max * min(1.0, self.kp_reactive * (0.98 - v_min))
            elif v_max > 1.02:
                action[i] = -q_max * min(1.0, self.kp_reactive * (v_max - 1.02))
            else:
                action[i] = 0.0
        
        # Proportional capacitor control
        if v_min < 0.97:
            # Need voltage support
            cap_signal = self.kp_cap * (0.97 - v_min)
            action[10] = min(0.3, cap_signal)
            action[11] = min(0.2, cap_signal * 0.7)
        else:
            # No capacitors needed
            action[10] = 0.0
            action[11] = 0.0
        
        # Proportional OLTC control
        if v_min < 0.96:
            # Boost voltage
            tap_adjust = self.kp_oltc * (0.96 - v_min)
            action[12] = max(0.9, 1.0 - tap_adjust)
        elif v_max > 1.04:
            # Reduce voltage
            tap_adjust = self.kp_oltc * (v_max - 1.04)
            action[12] = min(1.1, 1.0 + tap_adjust)
        else:
            action[12] = 1.0
        
        return action


class L3_PIControl:
    """Level 3: Proportional-Integral (PI) control with error integration."""
    
    def __init__(self, env):
        self.env = env
        # PI gains
        self.kp = 5.0
        self.ki = 0.1
        # Integral terms
        self.integral_error = 0.0
        self.integral_limit = 2.0
        # Reference
        self.v_ref = 1.0
        # History for filtering
        self.v_history = deque(maxlen=5)
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Update history and filter
        self.v_history.append(v_avg)
        v_filtered = np.mean(self.v_history) if len(self.v_history) > 0 else v_avg
        
        # PI control
        error = self.v_ref - v_filtered
        self.integral_error = np.clip(
            self.integral_error + error * 0.1,  # dt = 0.1
            -self.integral_limit, 
            self.integral_limit
        )
        
        control_signal = self.kp * error + self.ki * self.integral_error
        
        # Renewable control with PI-based curtailment
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                if control_signal < -0.02:  # Voltage too high
                    curtailment = min(1.0, abs(control_signal) * 2)
                    action[i] = gen.p_pot * (1 - curtailment)
                else:
                    action[i] = gen.p_pot
        
        # PI-based reactive control
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            q_max = q_limits[i-5]
            action[i] = np.clip(control_signal * 2, -q_max, q_max)
        
        # Capacitor control based on integral term
        if self.integral_error < -0.5:  # Persistent low voltage
            action[10] = 0.3
            action[11] = 0.2
        elif self.integral_error > 0.5:  # Persistent high voltage
            action[10] = 0.0
            action[11] = 0.0
        else:
            # Proportional response
            cap_signal = max(0, -control_signal * 5)
            action[10] = min(0.3, cap_signal)
            action[11] = min(0.2, cap_signal * 0.7)
        
        # OLTC with deadband to prevent hunting
        if abs(error) > 0.02:  # Only act if error significant
            action[12] = np.clip(1.0 - control_signal * 10, 0.9, 1.1)
        else:
            action[12] = 1.0
        
        return action


class L4_MPCControl:
    """Level 4: Model Predictive Control with lookahead optimization."""
    
    def __init__(self, env):
        self.env = env
        self.horizon = 3  # Prediction horizon
        self.v_ref = 1.0
        # Weights for cost function
        self.w_voltage = 10.0
        self.w_control = 1.0
        self.w_change = 0.5
        # Previous action for rate limiting
        self.prev_action = None
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Current state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Simple MPC: predict future voltage based on current trend
        if hasattr(self, 'v_prev'):
            v_trend = v_avg - self.v_prev
        else:
            v_trend = 0
        self.v_prev = v_avg
        
        # Predicted future voltages
        v_future = [v_avg + v_trend * (i+1) * 0.5 for i in range(self.horizon)]
        
        # Optimization: minimize cost over horizon
        # Cost = voltage deviation + control effort + rate of change
        
        # Renewable control with predictive curtailment
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                # Predict if voltage will be too high
                if any(v > 1.045 for v in v_future) or v_max > 1.04:
                    # Preemptive curtailment
                    severity = max(0, max(v_future) - 1.04)
                    action[i] = gen.p_pot * max(0.3, 1 - severity * 10)
                else:
                    action[i] = gen.p_pot
        
        # Predictive reactive control
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        future_error = np.mean([self.v_ref - v for v in v_future])
        for i in range(5, 10):
            q_max = q_limits[i-5]
            # Anticipatory control
            action[i] = np.clip(future_error * 10, -q_max, q_max)
        
        # Capacitor scheduling based on predicted trajectory
        if v_min < 0.96 or any(v < 0.965 for v in v_future):
            action[10] = 0.25
            action[11] = 0.15
        elif v_max > 1.04 or any(v > 1.045 for v in v_future):
            action[10] = 0.0
            action[11] = 0.0
        else:
            # Optimal steady-state
            action[10] = 0.1
            action[11] = 0.05
        
        # OLTC with rate limiting
        desired_tap = 1.0 - future_error * 15
        if self.prev_action is not None:
            # Limit rate of change
            prev_tap = self.prev_action[12]
            max_change = 0.05
            desired_tap = np.clip(desired_tap, prev_tap - max_change, prev_tap + max_change)
        action[12] = np.clip(desired_tap, 0.9, 1.1)
        
        self.prev_action = action.copy()
        return action


class L5_HierarchicalMPCControl:
    """Level 5: Hierarchical MPC with multi-timescale optimization."""
    
    def __init__(self, env):
        self.env = env
        # Multi-timescale horizons
        self.fast_horizon = 2    # For reactive control
        self.slow_horizon = 5    # For tap/capacitor
        # Reference tracking
        self.v_ref = 1.0
        # Hierarchical weights
        self.w_global = 5.0      # System-wide objective
        self.w_local = 3.0       # Local voltage control
        self.w_loss = 1.0        # Loss minimization
        # State estimation
        self.state_buffer = deque(maxlen=10)
        self.load_forecast = 1.0
        # Coordination - start with safe defaults
        self.cap_schedule = [0.1, 0.05]  # Some capacitor support
        self.tap_schedule = 1.0
        self.update_counter = 0
        # Adaptive parameters for robustness
        self.emergency_mode = False
        self.last_v_avg = 1.0
        # Smoothing for OLTC to prevent oscillations
        self.tap_history = deque([1.0, 1.0, 1.0], maxlen=3)
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # State estimation
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        bus_powers = np.array([bus.p for bus in sim.buses.values()])
        
        state = {
            'v_avg': np.mean(voltages),
            'v_min': np.min(voltages),
            'v_max': np.max(voltages),
            'v_std': np.std(voltages),
            'p_total': np.sum(bus_powers),
            'voltages': voltages.copy()
        }
        self.state_buffer.append(state)
        
        # Detect rapid changes (industrial switching, faults)
        v_change = abs(state['v_avg'] - self.last_v_avg)
        self.last_v_avg = state['v_avg']
        
        # Emergency mode with hysteresis to prevent oscillation
        if not self.emergency_mode:
            # Enter emergency mode only for severe conditions
            if v_change > 0.03 or state['v_min'] < 0.93 or state['v_max'] > 1.07:
                self.emergency_mode = True
                self.update_counter = 0  # Force immediate update
        else:
            # Exit emergency mode when stabilized
            if v_change < 0.01 and 0.95 <= state['v_min'] <= state['v_max'] <= 1.05:
                self.emergency_mode = False
        
        # Load forecasting (simple moving average)
        if len(self.state_buffer) >= 3:
            recent_loads = [s['p_total'] for s in list(self.state_buffer)[-3:]]
            self.load_forecast = np.mean(recent_loads) * 1.1  # 10% safety margin
        
        # Hierarchical control
        self.update_counter += 1
        
        # Update frequency based on mode
        update_freq = 2 if self.emergency_mode else 5
        
        # Slow timescale: OLTC and capacitor scheduling
        # Always update in first few steps to establish control
        if self.update_counter < 3 or self.update_counter % update_freq == 0 or self.emergency_mode:
            self._update_slow_controls(state)
        
        # Fast timescale: Renewable and reactive power
        
        # Distributed renewable control with loss minimization
        total_renewable = 0
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                bus_id = gen.bus_id
                local_v = voltages[bus_id] if bus_id < len(voltages) else state['v_avg']
                
                # Emergency mode: adaptive renewable use
                if self.emergency_mode:
                    if state['v_max'] > 1.05:
                        # Scale curtailment based on severity
                        curtailment = min(0.8, 10 * (state['v_max'] - 1.05))
                        action[i] = gen.p_pot * (1 - curtailment)
                    elif state['v_min'] < 0.95:
                        action[i] = gen.p_pot  # Use all available
                    else:
                        # Moderate use to avoid triggering issues
                        action[i] = gen.p_pot * 0.9
                else:
                    # Normal operation
                    # Local voltage constraint
                    if local_v > 1.048:
                        local_limit = gen.p_pot * max(0, 2 - 20*(local_v - 1.048))
                    else:
                        local_limit = gen.p_pot
                    
                    # Global coordination
                    if state['v_max'] > 1.045:
                        global_limit = gen.p_pot * 0.7
                    else:
                        global_limit = gen.p_pot
                    
                    action[i] = min(local_limit, global_limit)
                
                total_renewable += action[i]
        
        # Coordinated reactive power dispatch
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        
        # Emergency reactive support
        if self.emergency_mode:
            for i in range(5, 10):
                q_max = q_limits[i-5]
                if state['v_min'] < 0.95:
                    action[i] = q_max  # Max reactive support
                elif state['v_max'] > 1.05:
                    action[i] = -q_max  # Absorb reactive
                else:
                    # Proportional to voltage deviation
                    v_error = 1.0 - state['v_avg']
                    action[i] = np.clip(v_error * 20, -q_max, q_max)
        else:
            # Normal coordinated dispatch
            v_targets = self._compute_voltage_targets(state)
            
            for i in range(5, 10):
                gen_id = 36 + (i-5)
                if gen_id in sim.devices:
                    bus_id = sim.devices[gen_id].bus_id
                    local_v = voltages[bus_id] if bus_id < len(voltages) else state['v_avg']
                    target_v = v_targets.get(bus_id, 1.0)
                    
                    # Local PI control to target
                    error = target_v - local_v
                    q_max = q_limits[i-5]
                    action[i] = np.clip(error * 15, -q_max, q_max)
        
        # Apply scheduled discrete controls
        action[10] = self.cap_schedule[0]
        action[11] = self.cap_schedule[1]
        
        # Smooth OLTC to prevent oscillations
        self.tap_history.append(self.tap_schedule)
        smoothed_tap = np.mean(self.tap_history)
        # Round to nearest valid tap position
        valid_taps = [0.9, 0.95, 1.0, 1.05, 1.1]
        action[12] = min(valid_taps, key=lambda x: abs(x - smoothed_tap))
        
        return action
    
    def _update_slow_controls(self, state):
        """Update slow timescale controls (OLTC and capacitors)."""
        if self.emergency_mode:
            # More measured response in emergency - avoid oscillations
            v_avg = state['v_avg']
            v_min = state['v_min']
            v_max = state['v_max']
            
            # Consider both min and max voltages to avoid oscillation
            if v_min < 0.94 and v_max < 1.02:
                # Pure undervoltage - boost moderately
                self.tap_schedule = 0.95  # Not too aggressive
                self.cap_schedule = [0.25, 0.2]
            elif v_max > 1.06 and v_min > 0.98:
                # Pure overvoltage - reduce moderately
                self.tap_schedule = 1.05  # Not too aggressive
                self.cap_schedule = [0.0, 0.0]
            elif v_avg < 0.98:
                # Average is low - slight boost
                self.tap_schedule = 0.98
                self.cap_schedule = [0.15, 0.1]
            elif v_avg > 1.02:
                # Average is high - slight reduction
                self.tap_schedule = 1.02
                self.cap_schedule = [0.0, 0.0]
            else:
                # Near normal - minimal intervention
                self.tap_schedule = 1.0
                self.cap_schedule = [0.05, 0.0]
        else:
            # Normal predictive scheduling
            expected_v_drop = 0.02 * (self.load_forecast / 3.7)  # Empirical model
            
            # OLTC scheduling - more aggressive for voltage support
            if state['v_min'] < 0.965:
                self.tap_schedule = 0.95
            elif state['v_max'] > 1.04:
                self.tap_schedule = 1.05
            elif state['v_avg'] < 0.985:
                self.tap_schedule = 0.98
            elif state['v_avg'] > 1.015:
                self.tap_schedule = 1.02
            else:
                self.tap_schedule = 1.0
            
            # Capacitor scheduling with better coordination
            if state['v_min'] < 0.97:
                self.cap_schedule = [0.25, 0.15]
            elif state['v_max'] > 1.04:
                self.cap_schedule = [0.0, 0.0]
            elif state['v_min'] < 0.98:
                self.cap_schedule = [0.15, 0.1]
            elif state['v_avg'] < 0.995:
                self.cap_schedule = [0.1, 0.05]
            else:
                self.cap_schedule = [0.05, 0.0]
    
    def _compute_voltage_targets(self, state):
        """Compute optimal voltage targets for each bus."""
        # Simple heuristic: buses with low voltage get boosted
        targets = {}
        for i, v in enumerate(state['voltages']):
            if v < 0.97:
                targets[i] = 1.0
            elif v > 1.03:
                targets[i] = 1.0
            else:
                targets[i] = 1.0
        return targets