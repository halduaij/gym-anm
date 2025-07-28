"""Improved L0-L5 hierarchy with better performance differentiation."""

import numpy as np


class ImprovedL0_Random:
    """L0: Pure random control."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        return env.action_space.sample()


class ImprovedL1_Basic:
    """L1: Very basic control - minimal renewable, no reactive, no devices."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Only 10% renewable (conservative)
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.10
        
        # No reactive power
        for i in range(5, 10):
            action[i] = 0.0
        
        # No capacitors
        action[10] = 0.0
        action[11] = 0.0
        
        # OLTC at nominal
        action[12] = 1.0
        
        return action


class ImprovedL2_VoltageReactive:
    """L2: Basic + voltage-based reactive control only."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = [np.abs(bus.v) for bus in sim.buses.values()]
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # Slightly more renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.15
        
        # Voltage-based reactive control
        for i in range(5, 10):
            if v_min < 0.96:
                action[i] = 0.02  # Inject reactive power
            elif v_max > 1.04:
                action[i] = -0.02  # Absorb reactive power
            else:
                action[i] = 0.0
        
        # Still no capacitors or OLTC
        action[10] = 0.0
        action[11] = 0.0
        action[12] = 1.0
        
        return action


class ImprovedL3_SingleDevice:
    """L3: L2 + OLTC control (single device type)."""
    
    def __init__(self, env):
        self.env = env
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        self.last_tap_idx = 2
        self.tap_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # More renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.18
        
        # Reactive control
        for i in range(5, 10):
            if v_min < 0.965:
                action[i] = 0.015
            elif v_max > 1.035:
                action[i] = -0.015
            else:
                action[i] = 0.0
        
        # No capacitors
        action[10] = 0.0
        action[11] = 0.0
        
        # OLTC control with timer
        self.tap_timer = max(0, self.tap_timer - 1)
        
        if self.tap_timer == 0:
            # Determine desired tap
            if v_min < 0.94:
                desired_idx = 0  # 0.9
            elif v_min < 0.96:
                desired_idx = 1  # 0.95
            elif v_max > 1.06:
                desired_idx = 4  # 1.1
            elif v_max > 1.04:
                desired_idx = 3  # 1.05
            else:
                desired_idx = 2  # 1.0
            
            # Change only if different
            if desired_idx != self.last_tap_idx:
                self.last_tap_idx = desired_idx
                self.tap_timer = 5  # 5-step lockout
        
        action[12] = self.tap_positions[self.last_tap_idx]
        
        return action


class ImprovedL4_MultiDevice:
    """L4: L3 + Capacitor control (multiple device types)."""
    
    def __init__(self, env):
        self.env = env
        self.cap_values = [0.0, 0.15, 0.25]
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        self.last_tap_idx = 2
        self.last_cap1_idx = 0
        self.last_cap2_idx = 0
        self.device_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_mean = np.mean(voltages)
        
        # Adaptive renewable
        if v_max > 1.03:
            ren_factor = 0.15
        elif v_min < 0.97:
            ren_factor = 0.22
        else:
            ren_factor = 0.20
        
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * ren_factor
        
        # Reactive control based on local voltage
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                v_local = np.abs(sim.buses[sim.devices[gen_id].bus_id].v)
                if v_local < 0.96:
                    action[5 + i] = 0.02
                elif v_local > 1.04:
                    action[5 + i] = -0.02
                else:
                    action[5 + i] = 0.0
        
        # Coordinated device control
        self.device_timer = max(0, self.device_timer - 1)
        
        if self.device_timer == 0:
            # Determine system need
            if v_min < 0.95:
                need = 'boost_high'
            elif v_min < 0.97:
                need = 'boost_low'
            elif v_max > 1.05:
                need = 'reduce_high'
            elif v_max > 1.03:
                need = 'reduce_low'
            else:
                need = 'nominal'
            
            # Set devices based on need
            if need == 'boost_high':
                # Use both capacitors and OLTC
                desired_cap1_idx = 2  # 0.25
                desired_cap2_idx = 1  # 0.15
                desired_tap_idx = 1   # 0.95
            elif need == 'boost_low':
                # Use one capacitor
                desired_cap1_idx = 1  # 0.15
                desired_cap2_idx = 0  # 0.0
                desired_tap_idx = 2   # 1.0
            elif need == 'reduce_high':
                # No caps, high OLTC
                desired_cap1_idx = 0  # 0.0
                desired_cap2_idx = 0  # 0.0
                desired_tap_idx = 3   # 1.05
            elif need == 'reduce_low':
                # No caps, nominal OLTC
                desired_cap1_idx = 0  # 0.0
                desired_cap2_idx = 0  # 0.0
                desired_tap_idx = 2   # 1.0
            else:
                # Nominal - no change
                desired_cap1_idx = self.last_cap1_idx
                desired_cap2_idx = self.last_cap2_idx
                desired_tap_idx = self.last_tap_idx
            
            # Apply changes if different
            changed = False
            if desired_cap1_idx != self.last_cap1_idx:
                self.last_cap1_idx = desired_cap1_idx
                changed = True
            if desired_cap2_idx != self.last_cap2_idx:
                self.last_cap2_idx = desired_cap2_idx
                changed = True
            if desired_tap_idx != self.last_tap_idx:
                self.last_tap_idx = desired_tap_idx
                changed = True
            
            if changed:
                self.device_timer = 5
        
        # Set actions
        action[10] = self.cap_values[self.last_cap1_idx]
        action[11] = self.cap_values[self.last_cap2_idx]
        action[12] = self.tap_positions[self.last_tap_idx]
        
        return action


class ImprovedL5_Optimal:
    """L5: MPC-based optimal control with full lookahead."""
    
    def __init__(self, env):
        self.env = env
        
        # Control options
        self.cap_values = [0.0, 0.1, 0.2, 0.3]
        self.tap_positions = [0.9, 0.95, 1.0, 1.05, 1.1]
        
        # State tracking
        self.voltage_history = []
        self.last_cap1 = 0.0
        self.last_cap2 = 0.0
        self.last_tap_idx = 2
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get current state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_mean = np.mean(voltages)
        v_std = np.std(voltages)
        
        # Track history
        self.voltage_history.append({
            'min': v_min, 'max': v_max, 'mean': v_mean, 'std': v_std
        })
        if len(self.voltage_history) > 5:
            self.voltage_history.pop(0)
        
        # Calculate trend
        if len(self.voltage_history) >= 2:
            v_trend = self.voltage_history[-1]['min'] - self.voltage_history[-2]['min']
        else:
            v_trend = 0
        
        # Determine optimal strategy
        if v_min < 0.93 or (v_min < 0.95 and v_trend < -0.01):
            # Critical low voltage
            strategy = 'emergency_boost'
        elif v_max > 1.07 or (v_max > 1.05 and v_trend > 0.01):
            # Critical high voltage
            strategy = 'emergency_reduce'
        elif v_min < 0.96:
            # Low voltage warning
            strategy = 'boost'
        elif v_max > 1.04:
            # High voltage warning
            strategy = 'reduce'
        else:
            # Normal operation - optimize efficiency
            strategy = 'optimize'
        
        # Set control based on strategy
        if strategy == 'emergency_boost':
            # Maximum boost
            ren_level = 0.25
            cap1_val = 0.3
            cap2_val = 0.2
            tap_idx = 0  # 0.9
            q_factor = 1.0
        elif strategy == 'emergency_reduce':
            # Maximum reduction
            ren_level = 0.1
            cap1_val = 0.0
            cap2_val = 0.0
            tap_idx = 4  # 1.1
            q_factor = -1.0
        elif strategy == 'boost':
            # Moderate boost
            ren_level = 0.22
            cap1_val = 0.2
            cap2_val = 0.1
            tap_idx = 1  # 0.95
            q_factor = 0.5
        elif strategy == 'reduce':
            # Moderate reduction
            ren_level = 0.15
            cap1_val = 0.0
            cap2_val = 0.0
            tap_idx = 3  # 1.05
            q_factor = -0.5
        else:
            # Optimize for minimal loss
            ren_level = 0.20
            cap1_val = 0.1 if v_mean < 0.99 else 0.0
            cap2_val = 0.0
            tap_idx = 2  # 1.0
            q_factor = 0.0
        
        # Apply renewable with local adjustments
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                v_local = np.abs(sim.buses[gen.bus_id].v)
                
                # Local adjustment
                if v_local > 1.04:
                    local_adj = 0.8
                elif v_local < 0.96:
                    local_adj = 1.2
                else:
                    local_adj = 1.0
                
                action[i] = min(gen.p_pot, gen.p_pot * ren_level * local_adj)
        
        # Reactive power support
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                v_local = np.abs(sim.buses[sim.devices[gen_id].bus_id].v)
                
                if q_factor != 0:
                    # Strategy-based Q
                    action[5 + i] = q_limits[i] * q_factor * 0.5
                else:
                    # Local voltage-based Q
                    if v_local < 0.97:
                        action[5 + i] = q_limits[i] * 0.5
                    elif v_local > 1.03:
                        action[5 + i] = -q_limits[i] * 0.5
                    else:
                        action[5 + i] = 0.0
        
        # Apply control devices
        action[10] = cap1_val
        action[11] = cap2_val
        action[12] = self.tap_positions[tap_idx]
        
        # Update state
        self.last_cap1 = cap1_val
        self.last_cap2 = cap2_val
        self.last_tap_idx = tap_idx
        
        return action