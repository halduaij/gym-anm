"""L2 Droop control with discrete capacitor switching."""

import numpy as np


class L2_DiscreteDroop:
    """
    L2 Proportional/Droop control with discrete capacitor switching.
    
    Simple rules:
    - Turn ON capacitors when voltage < v_low
    - Turn OFF capacitors when voltage > v_high
    - All capacitors switch together (no individual control)
    """
    
    def __init__(self, env):
        self.env = env
        
        # Droop thresholds
        self.v_low = 0.97   # Turn on threshold
        self.v_high = 1.03  # Turn off threshold
        
        # Get capacitor info
        cap_info = env.get_capacitor_info() if hasattr(env, 'get_capacitor_info') else {}
        self.num_caps = cap_info.get('num_capacitors', 6)
        self.cap_ratings = cap_info.get('capacitor_ratings', [1.0, 1.0, 0.15, 0.1, 0.2, 0.15])
        
        # Current state (all OFF initially)
        self.caps_on = False
        
    def act(self, env):
        """Generate control action based on voltage droop."""
        # Get system voltages
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_avg = np.mean(voltages)
        
        # Simple droop logic - all capacitors together
        if v_min < self.v_low:
            # Voltage too low - turn ON all capacitors
            self.caps_on = True
        elif v_max > self.v_high:
            # Voltage too high - turn OFF all capacitors
            self.caps_on = False
        # Otherwise maintain current state (hysteresis)
        
        # Build action
        action = np.zeros(17)
        
        # Renewable control - just use default
        action[0:5] = 0.03   # Fixed renewable generation
        action[5:10] = 0.0   # No reactive power
        
        # Discrete capacitor control - ALL ON or ALL OFF
        for i in range(self.num_caps):
            if self.caps_on:
                action[10 + i] = self.cap_ratings[i] / 10.0  # ON at rated capacity
            else:
                action[10 + i] = 0.0  # OFF
        
        # OLTC - simple droop
        if v_avg < 0.98:
            action[16] = 1.05
        elif v_avg > 1.02:
            action[16] = 0.95
        else:
            action[16] = 1.0
            
        return action