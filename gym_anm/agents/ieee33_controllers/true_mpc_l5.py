"""True MPC-like L5 controller with discrete capacitor control."""

import numpy as np
from collections import deque


class L5_TrueMPC:
    """
    True Model Predictive Control implementation for L5.
    
    Key features:
    - Discrete capacitor control (ON/OFF only)
    - Multi-step prediction horizon
    - Considers future system trajectory
    - Minimizes total cost over horizon
    - Switching constraints
    """
    
    def __init__(self, env, prediction_horizon=10):
        self.env = env
        self.prediction_horizon = prediction_horizon
        
        # Get system info
        cap_info = env.get_capacitor_info() if hasattr(env, 'get_capacitor_info') else {}
        self.num_caps = cap_info.get('num_capacitors', 6)
        self.cap_buses = cap_info.get('capacitor_buses', [8, 25, 6, 12, 17, 32])
        self.cap_ratings = cap_info.get('capacitor_ratings', [1.0, 1.0, 0.15, 0.1, 0.2, 0.15])
        
        # Discrete capacitor states (ON/OFF)
        self.cap_states = np.zeros(self.num_caps, dtype=int)  # 0=OFF, 1=ON
        
        # History for prediction
        self.voltage_history = deque(maxlen=20)
        self.load_history = deque(maxlen=20)
        
        # Switching constraints
        self.min_on_time = 5  # Minimum time steps before switching
        self.min_off_time = 5
        self.time_since_switch = np.zeros(self.num_caps)
        
        # Cost weights
        self.w_voltage = 100.0    # Voltage deviation cost
        self.w_switching = 1.0    # Switching cost
        self.w_losses = 0.1       # System losses (approximated)
        
        # Voltage reference
        self.v_ref = 1.0
        self.v_min = 0.95
        self.v_max = 1.05
        
    def act(self, env):
        """Generate control action using MPC approach."""
        # Get current system state
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min, v_max, v_avg = np.min(voltages), np.max(voltages), np.mean(voltages)
        
        # Update histories
        self.voltage_history.append(voltages)
        self.load_history.append(v_avg)  # Use average voltage as proxy for load
        
        # Predict future trajectory
        future_voltages = self._predict_future_trajectory()
        
        # Find optimal capacitor configuration over horizon
        best_config, best_cost = self._optimize_capacitor_config(
            voltages, future_voltages
        )
        
        # Apply switching constraints
        feasible_config = self._apply_switching_constraints(best_config)
        
        # Update states
        self.cap_states = feasible_config
        self._update_switch_timers(feasible_config)
        
        # Build action vector
        action = np.zeros(17)
        
        # Renewable control (simple for now - focus on capacitors)
        action[0:5] = 0.03   # Moderate renewable generation
        action[5:10] = 0.0   # No reactive from renewables
        
        # Discrete capacitor control - ON/OFF only!
        for i in range(self.num_caps):
            if self.cap_states[i] == 1:
                # ON - use rated capacity
                action[10 + i] = self.cap_ratings[i] / 10.0  # Convert to p.u.
            else:
                # OFF
                action[10 + i] = 0.0
        
        # OLTC control
        if v_min < 0.94:
            action[16] = 1.05  # Boost voltage
        elif v_max > 1.06:
            action[16] = 0.95  # Reduce voltage
        else:
            action[16] = 1.0   # Nominal
            
        return action
    
    def _predict_future_trajectory(self):
        """Predict future voltage trajectory based on history."""
        if len(self.voltage_history) < 3:
            # Not enough history - assume constant
            return [self.voltage_history[-1] if self.voltage_history else np.ones(33)] * self.prediction_horizon
        
        # Simple linear prediction based on recent trend
        recent_avg = [np.mean(v) for v in list(self.voltage_history)[-5:]]
        if len(recent_avg) >= 2:
            trend = recent_avg[-1] - recent_avg[-2]
        else:
            trend = 0
        
        future = []
        last_v = self.voltage_history[-1]
        
        for t in range(self.prediction_horizon):
            # Predict voltage change
            predicted_v = last_v + trend * (t + 1)
            # Clip to reasonable range
            predicted_v = np.clip(predicted_v, 0.9, 1.1)
            future.append(predicted_v)
            
        return future
    
    def _optimize_capacitor_config(self, current_voltages, future_voltages):
        """Find optimal capacitor configuration over prediction horizon."""
        best_config = self.cap_states.copy()
        best_cost = float('inf')
        
        # For true MPC, we'd solve this optimization properly
        # Here we'll use a greedy search over configurations
        
        # Test current configuration
        current_cost = self._evaluate_config_cost(
            self.cap_states, current_voltages, future_voltages
        )
        
        # Try flipping each capacitor
        for i in range(self.num_caps):
            # Skip if switching constraint violated
            if not self._can_switch(i):
                continue
                
            # Try flipping this capacitor
            test_config = self.cap_states.copy()
            test_config[i] = 1 - test_config[i]
            
            cost = self._evaluate_config_cost(
                test_config, current_voltages, future_voltages
            )
            
            if cost < best_cost:
                best_cost = cost
                best_config = test_config.copy()
        
        # Try combinations (limited search for computational efficiency)
        # Try turning on/off pairs of capacitors
        for i in range(self.num_caps - 1):
            for j in range(i + 1, self.num_caps):
                if not (self._can_switch(i) and self._can_switch(j)):
                    continue
                    
                # Try all 4 combinations
                for state_i in [0, 1]:
                    for state_j in [0, 1]:
                        test_config = self.cap_states.copy()
                        test_config[i] = state_i
                        test_config[j] = state_j
                        
                        cost = self._evaluate_config_cost(
                            test_config, current_voltages, future_voltages
                        )
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_config = test_config.copy()
        
        return best_config, best_cost
    
    def _evaluate_config_cost(self, config, current_voltages, future_voltages):
        """Evaluate cost of a capacitor configuration over horizon."""
        total_cost = 0
        
        # Voltage deviation cost (current)
        v_dev_current = self._estimate_voltage_with_config(current_voltages, config)
        voltage_cost = self.w_voltage * np.sum(np.maximum(0, v_dev_current - self.v_max)**2 + 
                                               np.maximum(0, self.v_min - v_dev_current)**2)
        total_cost += voltage_cost
        
        # Future voltage costs (discounted)
        discount = 0.9
        for t, future_v in enumerate(future_voltages):
            v_dev_future = self._estimate_voltage_with_config(future_v, config)
            future_cost = self.w_voltage * np.sum(np.maximum(0, v_dev_future - self.v_max)**2 + 
                                                  np.maximum(0, self.v_min - v_dev_future)**2)
            total_cost += (discount ** (t + 1)) * future_cost
        
        # Switching cost
        switches = np.sum(config != self.cap_states)
        switching_cost = self.w_switching * switches
        total_cost += switching_cost
        
        # Approximate loss cost (more capacitors = more losses in light load)
        if np.mean(current_voltages) > 0.98:  # Light load condition
            loss_penalty = self.w_losses * np.sum(config) * 10
            total_cost += loss_penalty
        
        return total_cost
    
    def _estimate_voltage_with_config(self, base_voltages, cap_config):
        """Estimate voltage impact of capacitor configuration."""
        # Simplified voltage estimation
        # Each capacitor boosts voltage proportionally to its rating
        voltage_boost = 0
        
        for i, is_on in enumerate(cap_config):
            if is_on:
                # Estimate voltage boost (simplified)
                # Larger capacitors have more impact
                boost = 0.01 * self.cap_ratings[i]  # ~1% per MVAr
                voltage_boost += boost
        
        # Apply boost with diminishing returns
        estimated_voltages = base_voltages + voltage_boost * (2 - base_voltages)
        
        return estimated_voltages
    
    def _can_switch(self, cap_idx):
        """Check if capacitor can switch based on timing constraints."""
        if self.cap_states[cap_idx] == 1:  # Currently ON
            return self.time_since_switch[cap_idx] >= self.min_on_time
        else:  # Currently OFF
            return self.time_since_switch[cap_idx] >= self.min_off_time
    
    def _apply_switching_constraints(self, desired_config):
        """Apply switching constraints to desired configuration."""
        feasible_config = desired_config.copy()
        
        for i in range(self.num_caps):
            if desired_config[i] != self.cap_states[i]:
                # Want to switch
                if not self._can_switch(i):
                    # Can't switch yet - keep current state
                    feasible_config[i] = self.cap_states[i]
                    
        return feasible_config
    
    def _update_switch_timers(self, new_config):
        """Update switching timers."""
        for i in range(self.num_caps):
            if new_config[i] == self.cap_states[i]:
                # No switch - increment timer
                self.time_since_switch[i] += 1
            else:
                # Switched - reset timer
                self.time_since_switch[i] = 0