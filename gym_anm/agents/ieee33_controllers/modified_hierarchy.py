"""Modified L0-L5 hierarchy with intentional performance variation."""

import numpy as np


class ModifiedL0_Random:
    """L0: Pure random control."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        return env.action_space.sample()


class ModifiedL1_Conservative:
    """L1: Ultra-conservative - uses almost nothing."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Only 20% renewable - heavy curtailment penalty
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.2
        
        # No reactive, no devices
        for i in range(5, 10):
            action[i] = 0.0
        action[10] = 0.0
        action[11] = 0.0
        action[12] = 1.0
        
        return action


class ModifiedL2_Wasteful:
    """L2: Wasteful reactive - uses too much Q (increases losses)."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # 60% renewable - still significant curtailment
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.6
        
        # Always use maximum reactive (wasteful, increases losses)
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            action[i] = q_limits[i-5]  # Max Q injection always
        
        # No devices
        action[10] = 0.0
        action[11] = 0.0
        action[12] = 1.0
        
        return action


class ModifiedL3_Aggressive:
    """L3: Over-aggressive OLTC - changes too frequently."""
    
    def __init__(self, env):
        self.env = env
        self.step_count = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # 80% renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.8
        
        # Moderate reactive
        voltages = [np.abs(bus.v) for bus in sim.buses.values()]
        v_min = np.min(voltages)
        
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            if v_min < 0.97:
                action[i] = q_limits[i-5] * 0.5
            else:
                action[i] = 0.0
        
        # No capacitors
        action[10] = 0.0
        action[11] = 0.0
        
        # Aggressive OLTC - changes every 2 steps (bad for equipment)
        self.step_count += 1
        if self.step_count % 2 == 0:
            if v_min < 0.98:
                action[12] = 0.95
            else:
                action[12] = 1.05
        else:
            action[12] = 1.0
        
        return action


class ModifiedL4_Uncoordinated:
    """L4: Uses all devices but poorly coordinated."""
    
    def __init__(self, env):
        self.env = env
        self.cap_on = True  # Oscillates capacitors
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # 90% renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.9
        
        # Reactive based on voltage
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            if v_min < 0.96:
                action[i] = q_limits[i-5] * 0.75
            else:
                action[i] = -q_limits[i-5] * 0.25  # Absorb when not needed
        
        # Oscillating capacitors (poor coordination)
        self.cap_on = not self.cap_on
        if self.cap_on:
            action[10] = 0.3
            action[11] = 0.2
        else:
            action[10] = 0.0
            action[11] = 0.0
        
        # OLTC based on capacitor state (poor coordination)
        if self.cap_on:
            action[12] = 1.05  # Wrong direction when caps on
        else:
            action[12] = 0.95
        
        return action


class ModifiedL5_Smart:
    """L5: Actually optimizes - uses all renewable and minimizes losses."""
    
    def __init__(self, env):
        self.env = env
        self.voltage_history = []
        self.last_caps = [0.0, 0.0]
        self.last_tap = 1.0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get current state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_mean = np.mean(voltages)
        
        # Track voltage trend
        self.voltage_history.append(v_min)
        if len(self.voltage_history) > 3:
            self.voltage_history.pop(0)
        
        # 1. Use 100% renewable to avoid curtailment penalty
        total_renewable = 0
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                # Use all renewable but adjust slightly for voltage
                if v_max > 1.045:
                    action[i] = gen.p_pot * 0.98
                else:
                    action[i] = gen.p_pot * 1.0  # Use ALL available
                total_renewable += action[i]
        
        # 2. Smart reactive - only when truly needed
        q_limits = [0.02, 0.02, 0.02, 0.04, 0.04]
        for i in range(5, 10):
            if v_min < 0.94:  # Emergency only
                action[i] = q_limits[i-5] * 0.5
            elif v_max > 1.06:  # Emergency only
                action[i] = -q_limits[i-5] * 0.5
            else:
                action[i] = 0.0  # Minimize reactive (reduces losses)
        
        # 3. Intelligent device coordination
        # Avoid frequent switching
        if len(self.voltage_history) >= 3:
            trend = self.voltage_history[-1] - self.voltage_history[0]
        else:
            trend = 0
        
        # Capacitor decisions with hysteresis
        cap1_new = self.last_caps[0]
        cap2_new = self.last_caps[1]
        tap_new = self.last_tap
        
        if total_renewable > 0.1:  # Day time
            # With renewable, be conservative
            if v_min < 0.96 and self.last_caps[0] == 0:
                cap1_new = 0.1
            elif v_min > 0.98 and self.last_caps[0] > 0:
                cap1_new = 0.0
            
            # OLTC only if caps insufficient
            if v_min < 0.95 and cap1_new > 0:
                tap_new = 0.95
            elif v_max > 1.05:
                tap_new = 1.05
                cap1_new = 0.0  # Turn off caps if reducing
            else:
                tap_new = 1.0
        else:  # Night time - more aggressive
            if v_min < 0.94:
                cap1_new = 0.2
                cap2_new = 0.15
                tap_new = 0.95
            elif v_min < 0.96:
                cap1_new = 0.15
                cap2_new = 0.1
                tap_new = 0.95 if trend < 0 else 1.0
            elif v_min < 0.98:
                cap1_new = 0.1
                cap2_new = 0.0
                tap_new = 1.0
            else:
                # Good voltage - turn off gradually
                if self.last_caps[1] > 0:
                    cap2_new = 0.0
                elif self.last_caps[0] > 0:
                    cap1_new = 0.0
                tap_new = 1.0
        
        # Apply actions
        action[10] = cap1_new
        action[11] = cap2_new
        action[12] = tap_new
        
        # Update memory
        self.last_caps = [cap1_new, cap2_new]
        self.last_tap = tap_new
        
        return action
