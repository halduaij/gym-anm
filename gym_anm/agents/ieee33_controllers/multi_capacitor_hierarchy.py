"""Controller hierarchy adapted for multiple capacitors (6 instead of 2)."""

import numpy as np
from collections import deque


class L2_ProportionalControl_MultiCap:
    """L2 Proportional control adapted for 6 capacitors.
    
    Simple approach: All capacitors follow same proportional rule.
    This will be suboptimal as it doesn't consider capacitor locations.
    """
    
    def __init__(self, env):
        self.env = env
        self.kp_renewable = 5.0
        self.kp_reactive = 2.0
        self.kp_cap = 3.0
        self.kp_oltc = 5.0
        self.v_ref = 1.0
        
        # Get capacitor info
        cap_info = env.get_capacitor_info()
        self.num_caps = cap_info['num_capacitors']
        self.cap_buses = cap_info['capacitor_buses']
        self.cap_ratings = np.array(cap_info['capacitor_ratings']) / env.simulator.baseMVA
        
    def act(self, env):
        # 17 actions: 5 P, 5 Q, 6 capacitors, 1 OLTC
        action = np.zeros(17)
        sim = env.unwrapped.simulator
        
        # Get system voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Use worst-case voltage
        if abs(v_max - 1.05) > abs(0.95 - v_min):
            v_error = 1.0 - v_max
        else:
            v_error = 1.0 - v_min
        
        # Renewable P control
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                if v_max > 1.045:
                    curtailment = min(0.7, self.kp_renewable * (v_max - 1.045))
                    action[i] = gen.p_pot * (1 - curtailment)
                else:
                    action[i] = gen.p_pot
        
        # Reactive power control
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            q_max = q_limits[i-5]
            if v_min < 0.98:
                action[i] = q_max * min(1.0, self.kp_reactive * (0.98 - v_min))
            elif v_max > 1.02:
                action[i] = -q_max * min(1.0, self.kp_reactive * (v_max - 1.02))
            else:
                action[i] = 0.0
        
        # Simple proportional capacitor control - ALL SAME
        if v_min < 0.97:
            # All capacitors follow same rule (suboptimal!)
            cap_signal = self.kp_cap * (0.97 - v_min)
            for i in range(6):
                action[10 + i] = min(self.cap_ratings[i], cap_signal * self.cap_ratings[i])
        else:
            # All capacitors off
            for i in range(6):
                action[10 + i] = 0.0
        
        # OLTC control
        if v_min < 0.96:
            tap_adjust = self.kp_oltc * (0.96 - v_min)
            action[16] = max(0.9, 1.0 - tap_adjust)
        elif v_max > 1.04:
            tap_adjust = self.kp_oltc * (v_max - 1.04)
            action[16] = min(1.1, 1.0 + tap_adjust)
        else:
            action[16] = 1.0
        
        return action


class L5_HierarchicalMPC_MultiCap:
    """L5 Hierarchical MPC with intelligent multi-capacitor coordination.
    
    Advanced features:
    - Location-aware capacitor scheduling
    - Coordinated dispatch based on voltage profiles
    - Predictive staging of capacitors
    - Loss minimization through optimal placement
    """
    
    def __init__(self, env):
        self.env = env
        # Multi-timescale horizons
        self.fast_horizon = 2
        self.slow_horizon = 5
        self.v_ref = 1.0
        # Hierarchical weights
        self.w_global = 5.0
        self.w_local = 3.0
        self.w_loss = 1.0
        # State estimation
        self.state_buffer = deque(maxlen=10)
        self.load_forecast = 1.0
        
        # Get capacitor info
        cap_info = env.get_capacitor_info()
        self.num_caps = cap_info['num_capacitors']
        self.cap_buses = cap_info['capacitor_buses']
        self.cap_ratings = np.array(cap_info['capacitor_ratings']) / env.simulator.baseMVA
        
        # Capacitor scheduling - individual control
        self.cap_schedule = np.zeros(self.num_caps)
        self.tap_schedule = 1.0
        self.update_counter = 0
        
        # Adaptive parameters
        self.emergency_mode = False
        self.last_v_avg = 1.0
        self.tap_history = deque([1.0, 1.0, 1.0], maxlen=3)
        
        # Capacitor usage history for coordination
        self.cap_usage_history = deque(maxlen=20)
        
    def act(self, env):
        action = np.zeros(17)
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
            'voltages': voltages.copy(),
            'v_profile': self._analyze_voltage_profile(voltages)
        }
        self.state_buffer.append(state)
        
        # Detect rapid changes
        v_change = abs(state['v_avg'] - self.last_v_avg)
        self.last_v_avg = state['v_avg']
        
        # Emergency mode with hysteresis
        if not self.emergency_mode:
            if v_change > 0.03 or state['v_min'] < 0.93 or state['v_max'] > 1.07:
                self.emergency_mode = True
                self.update_counter = 0
        else:
            if v_change < 0.01 and 0.95 <= state['v_min'] <= state['v_max'] <= 1.05:
                self.emergency_mode = False
        
        # Load forecasting
        if len(self.state_buffer) >= 3:
            recent_loads = [s['p_total'] for s in list(self.state_buffer)[-3:]]
            self.load_forecast = np.mean(recent_loads) * 1.1
        
        # Hierarchical control
        self.update_counter += 1
        update_freq = 2 if self.emergency_mode else 5
        
        # Update slow controls
        if self.update_counter < 3 or self.update_counter % update_freq == 0 or self.emergency_mode:
            self._update_slow_controls_multicap(state)
        
        # Fast timescale: Renewable control
        total_renewable = 0
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                bus_id = gen.bus_id
                local_v = voltages[bus_id] if bus_id < len(voltages) else state['v_avg']
                
                if self.emergency_mode:
                    if state['v_max'] > 1.05:
                        curtailment = min(0.8, 10 * (state['v_max'] - 1.05))
                        action[i] = gen.p_pot * (1 - curtailment)
                    elif state['v_min'] < 0.95:
                        action[i] = gen.p_pot
                    else:
                        action[i] = gen.p_pot * 0.9
                else:
                    # Normal operation
                    if local_v > 1.048:
                        local_limit = gen.p_pot * max(0, 2 - 20*(local_v - 1.048))
                    else:
                        local_limit = gen.p_pot
                    
                    if state['v_max'] > 1.045:
                        global_limit = gen.p_pot * 0.7
                    else:
                        global_limit = gen.p_pot
                    
                    action[i] = min(local_limit, global_limit)
                
                total_renewable += action[i]
        
        # Coordinated reactive power
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        
        if self.emergency_mode:
            for i in range(5, 10):
                q_max = q_limits[i-5]
                if state['v_min'] < 0.95:
                    action[i] = q_max
                elif state['v_max'] > 1.05:
                    action[i] = -q_max
                else:
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
                    
                    error = target_v - local_v
                    q_max = q_limits[i-5]
                    action[i] = np.clip(error * 15, -q_max, q_max)
        
        # Apply scheduled capacitor controls
        for i in range(6):
            action[10 + i] = self.cap_schedule[i]
        
        # Smooth OLTC
        self.tap_history.append(self.tap_schedule)
        smoothed_tap = np.mean(self.tap_history)
        valid_taps = [0.9, 0.95, 1.0, 1.05, 1.1]
        action[16] = min(valid_taps, key=lambda x: abs(x - smoothed_tap))
        
        return action
    
    def _analyze_voltage_profile(self, voltages):
        """Analyze voltage profile to identify problem areas."""
        profile = {
            'low_v_buses': np.where(voltages < 0.97)[0].tolist(),
            'high_v_buses': np.where(voltages > 1.03)[0].tolist(),
            'critical_low': np.where(voltages < 0.95)[0].tolist(),
            'critical_high': np.where(voltages > 1.05)[0].tolist(),
            'gradient': np.gradient(voltages)
        }
        return profile
    
    def _update_slow_controls_multicap(self, state):
        """Update OLTC and multiple capacitors with intelligent coordination."""
        v_profile = state['v_profile']
        
        if self.emergency_mode:
            # Emergency response
            v_min = state['v_min']
            v_max = state['v_max']
            
            # OLTC scheduling
            if v_min < 0.94 and v_max < 1.02:
                self.tap_schedule = 0.95
            elif v_max > 1.06 and v_min > 0.98:
                self.tap_schedule = 1.05
            elif state['v_avg'] < 0.98:
                self.tap_schedule = 0.98
            elif state['v_avg'] > 1.02:
                self.tap_schedule = 1.02
            else:
                self.tap_schedule = 1.0
            
            # Emergency capacitor dispatch - all available for undervoltage
            if v_min < 0.95:
                self.cap_schedule = self.cap_ratings * 0.9  # Near full
            else:
                self.cap_schedule = np.zeros(self.num_caps)
        
        else:
            # Normal intelligent scheduling
            
            # OLTC based on average and forecast
            expected_v_drop = 0.02 * (self.load_forecast / 3.7)
            
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
            
            # Intelligent capacitor scheduling based on location
            self.cap_schedule = np.zeros(self.num_caps)
            
            # Prioritize capacitors near low voltage buses
            for i, cap_bus in enumerate(self.cap_buses):
                local_v = state['voltages'][cap_bus] if cap_bus < len(state['voltages']) else state['v_avg']
                
                # Local voltage-based decision
                if local_v < 0.96:
                    self.cap_schedule[i] = self.cap_ratings[i] * 0.8
                elif local_v < 0.97:
                    self.cap_schedule[i] = self.cap_ratings[i] * 0.5
                elif local_v < 0.98:
                    self.cap_schedule[i] = self.cap_ratings[i] * 0.3
                else:
                    self.cap_schedule[i] = 0.0
                
                # Global coordination - reduce if system voltage is high
                if state['v_max'] > 1.04:
                    self.cap_schedule[i] *= 0.5
                elif state['v_max'] > 1.045:
                    self.cap_schedule[i] = 0.0
            
            # Loss minimization - prefer upstream capacitors when multiple needed
            if np.sum(self.cap_schedule > 0) > 3:
                # Sort by bus number (upstream first)
                cap_priorities = np.argsort(self.cap_buses)
                total_needed = np.sum(self.cap_schedule)
                
                # Redistribute to minimize losses
                new_schedule = np.zeros(self.num_caps)
                allocated = 0
                
                for idx in cap_priorities:
                    if allocated < total_needed:
                        allocation = min(self.cap_ratings[idx], total_needed - allocated)
                        new_schedule[idx] = allocation
                        allocated += allocation
                
                self.cap_schedule = new_schedule * 0.8  # Safety factor
        
        # Record usage for analysis
        self.cap_usage_history.append(self.cap_schedule.copy())
    
    def _compute_voltage_targets(self, state):
        """Compute optimal voltage targets considering capacitor locations."""
        targets = {}
        
        # Set targets based on capacitor locations
        for i, cap_bus in enumerate(self.cap_buses):
            if self.cap_schedule[i] > 0:
                # If capacitor is on, local bus should be slightly higher
                targets[cap_bus] = 1.01
            else:
                targets[cap_bus] = 1.0
        
        # Other buses based on voltage
        for i, v in enumerate(state['voltages']):
            if i not in targets:
                if v < 0.97:
                    targets[i] = 1.0
                elif v > 1.03:
                    targets[i] = 1.0
                else:
                    targets[i] = 1.0
        
        return targets