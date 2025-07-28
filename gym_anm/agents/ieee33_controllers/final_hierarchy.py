"""Final L0-L5 hierarchy with proper performance differentiation."""

import numpy as np


class FinalL0_Random:
    """L0: Pure random control."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        return env.action_space.sample()


class FinalL1_Minimal:
    """L1: Minimal control - 50% renewable only, no devices."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Only 50% renewable (significant curtailment)
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.5
        
        # No reactive power
        for i in range(5, 10):
            action[i] = 0.0
        
        # No devices
        action[10] = 0.0
        action[11] = 0.0
        action[12] = 1.0
        
        return action


class FinalL2_Reactive:
    """L2: 70% renewable + reactive control."""
    
    def __init__(self, env):
        self.env = env
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = [np.abs(bus.v) for bus in sim.buses.values()]
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # 70% renewable (some curtailment)
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.7
        
        # Basic reactive control
        for i in range(5, 10):
            if v_min < 0.96:
                action[i] = 0.015
            elif v_max > 1.04:
                action[i] = -0.015
            else:
                action[i] = 0.0
        
        # No devices
        action[10] = 0.0
        action[11] = 0.0
        action[12] = 1.0
        
        return action


class FinalL3_SingleDevice:
    """L3: 85% renewable + reactive + OLTC."""
    
    def __init__(self, env):
        self.env = env
        self.last_tap = 1.0
        self.tap_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # 85% renewable
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.85
        
        # Reactive control
        for i in range(5, 10):
            if v_min < 0.965:
                action[i] = 0.01
            elif v_max > 1.035:
                action[i] = -0.01
            else:
                action[i] = 0.0
        
        # No capacitors
        action[10] = 0.0
        action[11] = 0.0
        
        # OLTC with timer
        self.tap_timer = max(0, self.tap_timer - 1)
        if self.tap_timer == 0:
            if v_min < 0.95:
                new_tap = 0.95
            elif v_max > 1.05:
                new_tap = 1.05
            else:
                new_tap = 1.0
            
            if new_tap != self.last_tap:
                self.last_tap = new_tap
                self.tap_timer = 5
        
        action[12] = self.last_tap
        
        return action


class FinalL4_MultiDevice:
    """L4: 95% renewable + all devices with simple logic."""
    
    def __init__(self, env):
        self.env = env
        self.cap_state = [0.0, 0.0]
        self.last_tap = 1.0
        self.device_timer = 0
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get voltages
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        
        # 95% renewable (minimal curtailment)
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                action[i] = gen.p_pot * 0.95
        
        # Reactive control
        for i in range(5, 10):
            if v_min < 0.97:
                action[i] = 0.01
            elif v_max > 1.03:
                action[i] = -0.01
            else:
                action[i] = 0.0
        
        # Device control with timer
        self.device_timer = max(0, self.device_timer - 1)
        if self.device_timer == 0:
            # Simple logic
            if v_min < 0.95:
                self.cap_state = [0.15, 0.1]
                self.last_tap = 0.95
            elif v_min < 0.97:
                self.cap_state = [0.1, 0.0]
                self.last_tap = 1.0
            elif v_max > 1.05:
                self.cap_state = [0.0, 0.0]
                self.last_tap = 1.05
            else:
                self.cap_state = [0.0, 0.0]
                self.last_tap = 1.0
            
            self.device_timer = 5
        
        action[10] = self.cap_state[0]
        action[11] = self.cap_state[1]
        action[12] = self.last_tap
        
        return action


class FinalL5_Optimal:
    """L5: True optimal control - 100% renewable + loss minimization."""
    
    def __init__(self, env):
        self.env = env
        self.voltage_history = []
        
    def act(self, env):
        action = np.zeros(13)
        sim = env.unwrapped.simulator
        
        # Get state
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_mean = np.mean(voltages)
        
        # Track history
        self.voltage_history.append({'min': v_min, 'max': v_max, 'mean': v_mean})
        if len(self.voltage_history) > 5:
            self.voltage_history.pop(0)
        
        # 1. Use ALL renewable (no curtailment penalty)
        total_renewable = 0
        for i in range(5):
            gen_id = 36 + i
            if gen_id in sim.devices:
                gen = sim.devices[gen_id]
                if gen.p_pot > 0:
                    # Adjust slightly if voltage too high
                    if v_max > 1.045:
                        factor = 0.95
                    else:
                        factor = 1.0
                    action[i] = gen.p_pot * factor
                    total_renewable += action[i]
        
        # 2. Minimal reactive (increases losses)
        if v_min < 0.94:
            # Emergency only
            for i in range(5, 10):
                action[i] = 0.01
        else:
            for i in range(5, 10):
                action[i] = 0.0
        
        # 3. Optimal devices for loss minimization
        # Key: Keep voltage near 1.0 to minimize I²R losses
        
        if total_renewable > 0.1:
            # Day time with renewable
            if v_min < 0.97:
                cap1 = 0.1
                cap2 = 0.0
                tap = 1.0
            elif v_max > 1.04:
                cap1 = 0.0
                cap2 = 0.0
                tap = 1.05
            else:
                # Good range
                cap1 = 0.0
                cap2 = 0.0
                tap = 1.0
        else:
            # Evening/night without renewable
            # More aggressive voltage support
            if v_min < 0.95:
                cap1 = 0.15
                cap2 = 0.1
                tap = 0.95
            elif v_min < 0.97:
                cap1 = 0.1
                cap2 = 0.05
                tap = 1.0
            elif v_min < 0.98:
                cap1 = 0.05
                cap2 = 0.0
                tap = 1.0
            else:
                # Voltage OK
                cap1 = 0.0
                cap2 = 0.0
                tap = 1.0
        
        # Fine tune for voltage spread
        if v_max - v_min > 0.08 and v_mean < 0.99:
            tap = 0.95
        
        action[10] = cap1
        action[11] = cap2
        action[12] = tap
        
        return action