"""Corrected L0-L5 hierarchy with proper discrete control understanding."""

import numpy as np


class CorrectedL0_Random:
    """L0: Pure random control."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        return env.action_space.sample()


class CorrectedL1_Basic:
    """L1: Basic discrete control with fixed patterns."""
    
    def __init__(self, env):
        self.env = env
        # Define discrete values
        self.cap_on_value = 0.5   # Base value, will be scaled down
        self.cap_off_value = 0.0  # 0 MVAr when OFF
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Fixed renewable at 20%
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.2  # Actions are in p.u., not MW!
        
        # No reactive power from generators
        for i in range(5, 10):
            action[i] = 0.0
        
        # Capacitors OFF
        action[10] = self.cap_off_value
        action[11] = self.cap_off_value
        
        # OLTC at nominal tap
        action[12] = self.tap_positions[2]  # 1.0
        
        return action


class CorrectedL2_VoltageThreshold:
    """L2: Voltage-based discrete switching."""
    
    def __init__(self, env):
        self.env = env
        self.cap_on_value = 0.2  # Reduced to avoid overvoltage
        self.cap_off_value = 0.0
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        # Add state memory for hysteresis
        self.last_cap_state = [False, False]
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = [np.abs(bus.v) for bus in sim.buses.values()]
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Renewable based on voltage
        if v_max > 1.04:
            renewable_factor = 0.15
        elif v_min < 0.96:
            renewable_factor = 0.25
        else:
            renewable_factor = 0.22
        
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * renewable_factor  # Actions in p.u.
        
        # Simple Q support
        for i in range(5, 10):
            if v_min < 0.97:
                action[i] = 0.01
            elif v_max > 1.03:
                action[i] = -0.01
            else:
                action[i] = 0.0
        
        # Discrete capacitor control with hysteresis
        # Cap 1: Hysteresis to prevent oscillation
        if not self.last_cap_state[0]:  # Currently OFF
            if v_min < 0.96 and v_max < 1.02:
                action[10] = self.cap_on_value
                self.last_cap_state[0] = True
            else:
                action[10] = self.cap_off_value
        else:  # Currently ON
            if v_min > 0.98 or v_max > 1.04:
                action[10] = self.cap_off_value
                self.last_cap_state[0] = False
            else:
                action[10] = self.cap_on_value
        
        # Cap 2: More conservative
        if not self.last_cap_state[1]:  # Currently OFF
            if v_min < 0.955 and v_max < 1.01:
                action[11] = self.cap_on_value * 0.5
                self.last_cap_state[1] = True
            else:
                action[11] = self.cap_off_value
        else:  # Currently ON
            if v_min > 0.975 or v_max > 1.03:
                action[11] = self.cap_off_value
                self.last_cap_state[1] = False
            else:
                action[11] = self.cap_on_value * 0.5
        
        # Discrete OLTC - avoid extreme positions with capacitors
        # If capacitors are ON, be conservative with OLTC
        caps_active = self.last_cap_state[0] or self.last_cap_state[1]
        
        if caps_active:
            # With capacitors, only use mild OLTC
            if v_min < 0.94:
                tap_idx = 1  # 0.95 instead of 0.9
            elif v_max > 1.06:
                tap_idx = 3  # 1.05 instead of 1.1
            else:
                tap_idx = 2  # 1.0 - nominal
        else:
            # Without capacitors, can use full range
            if v_min < 0.93:
                tap_idx = 0  # 0.9 - only for emergency
            elif v_min < 0.96:
                tap_idx = 1  # 0.95
            elif v_max > 1.07:
                tap_idx = 4  # 1.1 - only for emergency
            elif v_max > 1.04:
                tap_idx = 3  # 1.05
            else:
                tap_idx = 2  # 1.0
        
        action[12] = self.tap_positions[tap_idx]
        
        return action


class CorrectedL3_Coordinated:
    """L3: Coordinated discrete control with system awareness."""
    
    def __init__(self, env):
        self.env = env
        self.cap_on_value = 0.15  # Very conservative
        self.cap_off_value = 0.0
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        # State tracking
        self.cap_state = [False, False]
        self.last_tap_idx = 2
        self.action_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # State analysis
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_mean = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_std = np.std(voltages)
        
        # Adaptive renewable
        margin = min(v_min - 0.95, 1.05 - v_max)
        if margin < 0.01:
            base_renewable = 0.15
        elif margin < 0.02:
            base_renewable = 0.20
        else:
            base_renewable = 0.24
        
        # Distribute based on local voltage
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                v_local = np.abs(sim.buses[gen.bus_id].v)
                
                if v_local > 1.035:
                    local_factor = 0.7
                elif v_local < 0.965:
                    local_factor = 1.2
                else:
                    local_factor = 1.0
                
                action[i] = gen.p_pot * base_renewable * local_factor
                action[i] = min(action[i], gen.p_pot)  # Cap at potential
        
        # Reactive power coordination
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                v_local = np.abs(sim.buses[sim.devices[gen_id].bus_id].v)
                
                if v_local < 0.97:
                    action[5 + i] = q_limits[i] * 0.5
                elif v_local > 1.03:
                    action[5 + i] = -q_limits[i] * 0.5
                else:
                    action[5 + i] = 0.0
        
        # Update action timer
        self.action_timer = max(0, self.action_timer - 1)
        
        # Coordinated discrete capacitors with lockout timer
        if self.action_timer == 0:
            # Determine desired capacitor state
            if v_min < 0.95 and v_max < 1.01:
                # Need boost
                desired_caps = [True, True]
            elif v_min < 0.96 and v_max < 1.02:
                # Mild boost
                desired_caps = [True, False]
            elif v_max > 1.04 or (v_max > 1.03 and v_mean > 1.01):
                # Too high
                desired_caps = [False, False]
            else:
                # Keep current state (hysteresis)
                desired_caps = self.cap_state
            
            # Apply changes only if different
            if desired_caps != self.cap_state:
                self.cap_state = desired_caps
                self.action_timer = 5  # Lockout for 5 steps
        
        # Set capacitor actions
        action[10] = self.cap_on_value if self.cap_state[0] else self.cap_off_value
        action[11] = self.cap_on_value * 0.5 if self.cap_state[1] else self.cap_off_value
        
        # Discrete OLTC coordinated with capacitors
        # Only change OLTC if capacitors alone aren't sufficient
        if self.action_timer == 0:  # Only when not in capacitor lockout
            if v_min < 0.94:
                desired_tap_idx = 0  # 0.9 - emergency
            elif v_min < 0.95 and not any(self.cap_state):
                desired_tap_idx = 1  # 0.95 - only if caps are off
            elif v_max > 1.06:
                desired_tap_idx = 4  # 1.1 - emergency
            elif v_max > 1.05 and not any(self.cap_state):
                desired_tap_idx = 3  # 1.05 - only if caps are off
            else:
                desired_tap_idx = 2  # 1.0 - nominal
            
            # Only change if different
            if desired_tap_idx != self.last_tap_idx:
                self.last_tap_idx = desired_tap_idx
                self.action_timer = 10  # Longer lockout for OLTC
        
        action[12] = self.tap_positions[self.last_tap_idx]
        
        return action


class CorrectedL4_Predictive:
    """L4: Predictive control with state memory and switching constraints."""
    
    def __init__(self, env):
        self.env = env
        self.cap_on_value = 0.4
        self.cap_off_value = 0.0
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        # State memory
        self.voltage_history = []
        self.last_caps = [self.cap_off_value, self.cap_off_value]
        self.last_tap_idx = 2
        self.cap_switch_timer = [0, 0]
        self.tap_change_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Update state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        self.voltage_history.append(voltages)
        if len(self.voltage_history) > 5:
            self.voltage_history.pop(0)
        
        v_mean = np.mean(voltages)
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Calculate trend
        if len(self.voltage_history) >= 2:
            v_trend = np.mean(self.voltage_history[-1]) - np.mean(self.voltage_history[-2])
        else:
            v_trend = 0
        
        # Predictive renewable dispatch
        if v_trend > 0.005 and v_max > 1.02:
            base_renewable = 0.16
        elif v_trend < -0.005 and v_min < 0.98:
            base_renewable = 0.24
        else:
            base_renewable = 0.20
        
        # Smart distribution
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                v_local = np.abs(sim.buses[gen.bus_id].v)
                v_predicted = v_local + v_trend * 3
                
                if v_predicted > 1.04:
                    local_factor = 0.6
                elif v_predicted < 0.96:
                    local_factor = 1.3
                else:
                    local_factor = 1.0
                
                action[i] = gen.p_pot * base_renewable * local_factor
                action[i] = min(action[i], gen.p_pot)  # Cap at potential
        
        # Predictive Q control
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                v_local = np.abs(sim.buses[sim.devices[gen_id].bus_id].v)
                v_predicted = v_local + v_trend * 3
                
                if v_predicted < 0.96 or v_local < 0.965:
                    action[5 + i] = q_limits[i] * 0.6
                elif v_predicted > 1.04 or v_local > 1.035:
                    action[5 + i] = -q_limits[i] * 0.6
                else:
                    action[5 + i] = 0.0
        
        # Update timers
        self.cap_switch_timer[0] = max(0, self.cap_switch_timer[0] - 1)
        self.cap_switch_timer[1] = max(0, self.cap_switch_timer[1] - 1)
        self.tap_change_timer = max(0, self.tap_change_timer - 1)
        
        # Predictive capacitor switching with hysteresis
        # Cap 1
        if self.cap_switch_timer[0] == 0:
            if v_min < 0.96 and self.last_caps[0] == self.cap_off_value:
                action[10] = self.cap_on_value
                self.cap_switch_timer[0] = 5
            elif v_min > 0.975 and self.last_caps[0] == self.cap_on_value:
                action[10] = self.cap_off_value
                self.cap_switch_timer[0] = 5
            else:
                action[10] = self.last_caps[0]
        else:
            action[10] = self.last_caps[0]
        
        # Cap 2
        if self.cap_switch_timer[1] == 0:
            if v_min < 0.955 and self.last_caps[1] == self.cap_off_value:
                action[11] = self.cap_on_value
                self.cap_switch_timer[1] = 5
            elif v_min > 0.97 and self.last_caps[1] == self.cap_on_value:
                action[11] = self.cap_off_value
                self.cap_switch_timer[1] = 5
            else:
                action[11] = self.last_caps[1]
        else:
            action[11] = self.last_caps[1]
        
        self.last_caps = [action[10], action[11]]
        
        # Predictive OLTC with change limiting
        if self.tap_change_timer == 0:
            # Determine desired tap
            if v_min < 0.94 or (v_min < 0.95 and v_trend < -0.01):
                desired_idx = 0
            elif v_min < 0.96:
                desired_idx = 1
            elif v_max > 1.06 or (v_max > 1.05 and v_trend > 0.01):
                desired_idx = 4
            elif v_max > 1.04:
                desired_idx = 3
            else:
                desired_idx = 2
            
            # Only change if significant
            if abs(desired_idx - self.last_tap_idx) > 1 or v_min < 0.93 or v_max > 1.07:
                action[12] = self.tap_positions[desired_idx]
                self.last_tap_idx = desired_idx
                self.tap_change_timer = 10
            else:
                action[12] = self.tap_positions[self.last_tap_idx]
        else:
            action[12] = self.tap_positions[self.last_tap_idx]
        
        return action


class CorrectedL5_Optimal:
    """L5: MPC-based optimal control with proper coordination."""
    
    def __init__(self, env):
        self.env = env
        
        # MPC parameters
        self.prediction_horizon = 3
        self.control_horizon = 1
        
        # Discrete control values
        self.cap_values = [0.0, 0.2, 0.3]  # Discrete capacitor values
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        self.renewable_levels = [0.15, 0.20, 0.25]  # Discrete renewable levels
        
        # State tracking
        self.voltage_history = []
        self.action_history = []
        self.last_cap1 = 0.0
        self.last_cap2 = 0.0
        self.last_tap_idx = 2
        
        # Control constraints
        self.cap_switch_penalty = 0.001  # Very small penalty for switching
        self.tap_change_penalty = 0.005  # Small penalty for OLTC changes
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get current state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_mean = np.mean(voltages)
        
        # Update history
        self.voltage_history.append({'min': v_min, 'max': v_max, 'mean': v_mean})
        if len(self.voltage_history) > 10:
            self.voltage_history.pop(0)
        
        # MPC OPTIMIZATION (simplified for real-time control)
        best_action = None
        best_cost = float('inf')
        
        # Search over discrete action combinations
        for ren_level in self.renewable_levels:
            for cap1_val in self.cap_values:
                for cap2_val in self.cap_values:
                    for tap_idx in range(len(self.tap_positions)):
                        # Skip invalid combinations
                        if cap1_val + cap2_val > 0.5:  # Total capacitor limit
                            continue
                        
                        # Predict voltage impact
                        predicted_v_min, predicted_v_max = self._predict_voltage(
                            v_min, v_max, ren_level, cap1_val, cap2_val, 
                            self.tap_positions[tap_idx]
                        )
                        
                        # Calculate cost
                        cost = 0
                        
                        # Voltage violation cost (hard constraint)
                        if predicted_v_min < 0.95:
                            cost += 100 * (0.95 - predicted_v_min) ** 2
                        if predicted_v_max > 1.05:
                            cost += 100 * (predicted_v_max - 1.05) ** 2
                        
                        # Soft margin cost (prefer staying away from limits)
                        if predicted_v_min < 0.96:
                            cost += 1.0 * (0.96 - predicted_v_min) ** 2
                        if predicted_v_max > 1.04:
                            cost += 1.0 * (predicted_v_max - 1.04) ** 2
                        
                        # Voltage deviation from nominal (reduced weight)
                        cost += 0.05 * ((predicted_v_min + predicted_v_max)/2 - 1.0) ** 2
                        
                        # Switching costs
                        if cap1_val != self.last_cap1:
                            cost += self.cap_switch_penalty
                        if cap2_val != self.last_cap2:
                            cost += self.cap_switch_penalty
                        if tap_idx != self.last_tap_idx:
                            cost += self.tap_change_penalty
                        
                        # Control effort cost
                        cost += 0.01 * (cap1_val + cap2_val)
                        cost += 0.001 * abs(ren_level - 0.2)  # Prefer moderate renewable
                        
                        # Update best action
                        if cost < best_cost:
                            best_cost = cost
                            best_action = {
                                'ren': ren_level,
                                'cap1': cap1_val,
                                'cap2': cap2_val,
                                'tap_idx': tap_idx
                            }
        
        # Apply best action
        if best_action:
            # Renewable generation
            for i in range(5):
                gen_id = 36 + i
                if gen_id in sim.devices:
                    gen = sim.devices[gen_id]
                    # Check if renewable is available (not nighttime)
                    if gen.p_pot > 0:
                        action[i] = gen.p_pot * best_action['ren']
                    else:
                        action[i] = 0.0
            
            # Reactive power - only in emergencies
            for i in range(5, 10):
                if v_min < 0.94:
                    action[i] = 0.01
                elif v_max > 1.06:
                    action[i] = -0.01
                else:
                    action[i] = 0.0
            
            # Capacitors
            action[10] = best_action['cap1']
            action[11] = best_action['cap2']
            
            # OLTC
            action[12] = self.tap_positions[best_action['tap_idx']]
            
            # Update state
            self.last_cap1 = best_action['cap1']
            self.last_cap2 = best_action['cap2']
            self.last_tap_idx = best_action['tap_idx']
        else:
            # Fallback to safe defaults
            for i in range(5):
                gen_id = 36 + i
                if gen_id in sim.devices:
                    gen = sim.devices[gen_id]
                    action[i] = gen.p_pot * 0.2
            action[10] = 0.0
            action[11] = 0.0
            action[12] = 1.0
        
        return action
    
    def _predict_voltage(self, v_min, v_max, ren_level, cap1, cap2, tap):
        """Simplified voltage prediction model based on actual measurements."""
        # Current state affects prediction
        if v_min > 0.99 and v_max < 1.01:
            # System at nominal - load will cause drop
            base_change_min = -0.046
            base_change_max = 0.0
        else:
            # System already stressed - no additional drop
            base_change_min = 0.0
            base_change_max = 0.0
        
        # Capacitor impact (based on actual tests - very small!)
        cap_boost = (cap1 + cap2) * 0.005  # Only 0.001 p.u. per 0.2 MVAr
        
        # OLTC impact (INVERSE operation, from actual tests)
        if tap < 1.0:
            # tap < 1.0 INCREASES voltage
            if tap <= 0.95:
                oltc_boost = 0.046  # 0.95 tap gives ~0.046 boost
            else:
                oltc_boost = (1.0 - tap) * 0.92  # Linear interpolation
        else:
            # tap > 1.0 DECREASES voltage  
            if tap >= 1.05:
                oltc_boost = -0.050  # 1.05 tap gives ~0.050 drop
            else:
                oltc_boost = (1.0 - tap) * 1.0  # Linear interpolation
        
        # Renewable impact (negligible)
        ren_impact = 0.0
        
        # Predict new voltages
        pred_v_min = v_min + base_change_min + cap_boost + oltc_boost + ren_impact
        pred_v_max = v_max + base_change_max + cap_boost * 0.9 + oltc_boost * 0.95
        
        # Ensure physical limits
        pred_v_min = max(0.85, min(1.15, pred_v_min))
        pred_v_max = max(pred_v_min, min(1.15, pred_v_max))
        
        return pred_v_min, pred_v_max
