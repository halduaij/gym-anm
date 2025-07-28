"""L5 Hierarchical MPC modified to consider switching costs."""

import numpy as np
from collections import deque
from .multi_capacitor_hierarchy import L5_HierarchicalMPC_MultiCap


class L5_SwitchingAwareMPC(L5_HierarchicalMPC_MultiCap):
    """L5 Hierarchical MPC that explicitly considers switching costs.
    
    Modifications:
    - Tracks previous capacitor states
    - Adds hysteresis to prevent frequent switching
    - Considers switching cost in optimization
    - Implements minimum time between switches
    """
    
    def __init__(self, env, switching_cost=0.005):
        super().__init__(env)
        
        # Switching cost parameters
        self.switching_cost = switching_cost
        self.min_switch_interval = 10  # Minimum steps between switches
        
        # Tracking for switching optimization
        self.prev_cap_states = np.zeros(self.num_caps)
        self.last_switch_time = np.zeros(self.num_caps)
        self.time_since_switch = np.zeros(self.num_caps)
        
        # Hysteresis thresholds
        self.cap_on_threshold = 0.965   # Turn on if v < this
        self.cap_off_threshold = 0.985  # Turn off if v > this
        
        # Switching history for analysis
        self.switch_history = deque(maxlen=100)
        
    def _update_slow_controls_multicap(self, state):
        """Update OLTC and capacitors with switching cost awareness."""
        v_profile = state['v_profile']
        
        # Update time since last switch
        self.time_since_switch += 1
        
        if self.emergency_mode:
            # Emergency response - but still consider switching costs
            v_min = state['v_min']
            v_max = state['v_max']
            
            # OLTC scheduling (unchanged)
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
            
            # Emergency capacitor dispatch with switching awareness
            new_schedule = np.zeros(self.num_caps)
            
            if v_min < 0.93:  # Critical undervoltage
                # Turn on all capacitors regardless of cost
                new_schedule = self.cap_ratings * 0.9
            else:
                # Consider switching cost even in emergency
                for i in range(self.num_caps):
                    current_state = self.cap_schedule[i]
                    
                    # Only switch if benefit outweighs cost
                    if current_state < 0.1:  # Currently OFF
                        if v_min < 0.95 and self.time_since_switch[i] >= self.min_switch_interval:
                            new_schedule[i] = self.cap_ratings[i] * 0.8
                    else:  # Currently ON
                        if v_max > 1.05:
                            new_schedule[i] = 0.0
                        else:
                            new_schedule[i] = current_state  # Keep ON
            
            self.cap_schedule = new_schedule
            
        else:
            # Normal operation with full switching optimization
            
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
            
            # Intelligent capacitor scheduling with switching costs
            new_schedule = self.cap_schedule.copy()  # Start with current state
            
            for i, cap_bus in enumerate(self.cap_buses):
                local_v = state['voltages'][cap_bus] if cap_bus < len(state['voltages']) else state['v_avg']
                current_state = self.cap_schedule[i]
                
                # Calculate benefit of switching
                if current_state < 0.1:  # Currently OFF
                    # Consider turning ON
                    if local_v < self.cap_on_threshold:
                        # Estimate voltage improvement
                        v_improvement = self.cap_ratings[i] * 0.02  # Rough estimate
                        
                        # Expected benefit over next period
                        expected_benefit = v_improvement * self.slow_horizon
                        
                        # Switch if benefit exceeds cost and minimum interval passed
                        if (expected_benefit > self.switching_cost * self.cap_ratings[i] and 
                            self.time_since_switch[i] >= self.min_switch_interval):
                            new_schedule[i] = self.cap_ratings[i] * 0.7
                            self.time_since_switch[i] = 0
                            self._record_switch(i, 'ON', local_v)
                            
                else:  # Currently ON
                    # Consider turning OFF
                    if local_v > self.cap_off_threshold or state['v_max'] > 1.04:
                        # Only switch OFF if really needed or been ON for a while
                        if state['v_max'] > 1.045 or self.time_since_switch[i] > 50:
                            new_schedule[i] = 0.0
                            self.time_since_switch[i] = 0
                            self._record_switch(i, 'OFF', local_v)
            
            # Coordinated optimization - avoid switching multiple capacitors at once
            switches_planned = np.sum(np.abs(new_schedule - self.cap_schedule) > 0.05)
            
            if switches_planned > 2:  # Too many switches
                # Prioritize most critical switches
                voltage_errors = []
                for i, cap_bus in enumerate(self.cap_buses):
                    local_v = state['voltages'][cap_bus]
                    if new_schedule[i] != self.cap_schedule[i]:
                        error = abs(1.0 - local_v)
                        voltage_errors.append((i, error))
                
                # Sort by voltage error
                voltage_errors.sort(key=lambda x: x[1], reverse=True)
                
                # Only allow top 2 switches
                allowed_switches = [x[0] for x in voltage_errors[:2]]
                
                for i in range(self.num_caps):
                    if i not in allowed_switches and new_schedule[i] != self.cap_schedule[i]:
                        new_schedule[i] = self.cap_schedule[i]  # Cancel switch
            
            self.cap_schedule = new_schedule
        
        # Update previous states
        self.prev_cap_states = self.cap_schedule.copy()
        
        # Record usage for analysis
        self.cap_usage_history.append(self.cap_schedule.copy())
    
    def _record_switch(self, cap_idx, action, voltage):
        """Record switching event for analysis."""
        self.switch_history.append({
            'timestep': self.update_counter,
            'capacitor': cap_idx,
            'bus': self.cap_buses[cap_idx],
            'action': action,
            'voltage': voltage,
            'emergency': self.emergency_mode
        })
    
    def get_switching_stats(self):
        """Return switching statistics."""
        total_switches = len(self.switch_history)
        emergency_switches = sum(1 for s in self.switch_history if s['emergency'])
        
        return {
            'total_switches': total_switches,
            'emergency_switches': emergency_switches,
            'normal_switches': total_switches - emergency_switches,
            'avg_time_between_switches': np.mean(self.time_since_switch) if total_switches > 0 else 0
        }