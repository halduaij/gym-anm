"""Enhanced L5 controller that intelligently handles unequal capacitors and switching costs."""

import numpy as np


class L5_EnhancedSwitchingAware:
    """
    Enhanced L5 Hierarchical MPC with intelligent handling of:
    - Unequal capacitor sizes
    - Variable switching costs
    - Spatial voltage issues
    - Predictive load/renewable changes
    """
    
    def __init__(self, env):
        self.env = env
        
        # Get capacitor information
        cap_info = env.get_capacitor_info()
        self.num_caps = cap_info['num_capacitors']
        self.cap_buses = cap_info['capacitor_buses']
        self.cap_ratings = np.array(cap_info['capacitor_ratings'])
        self.switching_costs = np.array(cap_info.get('switching_costs', 
                                        0.01 * self.cap_ratings))
        
        # Categorize capacitors by size
        self.large_caps = [i for i, r in enumerate(self.cap_ratings) if r >= 2.0]
        self.medium_caps = [i for i, r in enumerate(self.cap_ratings) if 1.0 <= r < 2.0]
        self.small_caps = [i for i, r in enumerate(self.cap_ratings) if 0.4 <= r < 1.0]
        self.tiny_caps = [i for i, r in enumerate(self.cap_ratings) if r < 0.4]
        
        # Control parameters
        self.v_ref = 1.0
        self.v_deadband = 0.005
        self.emergency_threshold = 0.03
        self.prediction_horizon = 10
        
        # State tracking
        self.voltage_history = []
        self.load_history = []
        self.cap_switch_history = [0] * self.num_caps
        self.time_since_switch = [0] * self.num_caps
        self.emergency_mode = False
        self.oltc_position = 1.0
        
        # Minimum time between switches (varies by capacitor size)
        self.min_switch_intervals = [
            max(5, int(20 * (r / max(self.cap_ratings))))  # Large caps switch less
            for r in self.cap_ratings
        ]
        
        print(f"Enhanced L5 initialized with {self.num_caps} capacitors")
        print(f"  Large caps: {self.large_caps} at buses {[self.cap_buses[i] for i in self.large_caps]}")
        print(f"  Medium caps: {self.medium_caps} at buses {[self.cap_buses[i] for i in self.medium_caps]}")
        print(f"  Small caps: {self.small_caps} at buses {[self.cap_buses[i] for i in self.small_caps]}")
        print(f"  Tiny caps: {self.tiny_caps} at buses {[self.cap_buses[i] for i in self.tiny_caps]}")
        
    def act(self, env):
        """Generate control action with intelligent capacitor coordination."""
        # Get system state
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min, v_max = np.min(voltages), np.max(voltages)
        v_avg = np.mean(voltages)
        
        # Identify voltage problem locations
        problem_buses = [i for i, v in enumerate(voltages) if v < 0.97 or v > 1.03]
        
        # Update histories
        self.voltage_history.append(voltages)
        if len(self.voltage_history) > self.prediction_horizon:
            self.voltage_history.pop(0)
            
        # Estimate load level from voltage drop
        estimated_load = max(0, (1.02 - v_avg) * 20)
        self.load_history.append(estimated_load)
        if len(self.load_history) > self.prediction_horizon:
            self.load_history.pop(0)
            
        # Update switch timers
        for i in range(self.num_caps):
            self.time_since_switch[i] += 1
            
        # Initialize action
        action = np.zeros(17)
        
        # Renewable control (reactive priority for voltage)
        renewable_q = self._compute_renewable_q(voltages)
        action[5:10] = renewable_q
        
        # Capacitor control with size-aware logic
        cap_actions = self._compute_capacitor_actions(
            voltages, v_min, v_max, v_avg, problem_buses
        )
        action[10:16] = cap_actions
        
        # OLTC control
        oltc_action = self._compute_oltc_action(v_min, v_max, v_avg)
        action[16] = oltc_action
        
        return action
        
    def _compute_renewable_q(self, voltages):
        """Compute renewable reactive power for voltage support."""
        q_actions = np.zeros(5)
        
        # Use renewable Q for voltage support
        for i in range(5):
            bus_v = voltages[min(i*6, len(voltages)-1)]  # Approximate bus voltage
            if bus_v < 0.98:
                q_actions[i] = 0.3  # Inject vars
            elif bus_v > 1.02:
                q_actions[i] = -0.3  # Absorb vars
            else:
                q_actions[i] = 0.0
                
        return q_actions
        
    def _compute_capacitor_actions(self, voltages, v_min, v_max, v_avg, problem_buses):
        """Compute capacitor actions with intelligent size-based coordination."""
        cap_actions = np.zeros(self.num_caps)
        
        # Determine control mode
        if v_min < 0.95 or v_max > 1.05:
            self.emergency_mode = True
        elif v_min > 0.97 and v_max < 1.03:
            self.emergency_mode = False
            
        # Predict future voltage trend
        if len(self.voltage_history) >= 3:
            v_trend = np.mean(self.voltage_history[-1]) - np.mean(self.voltage_history[-3])
        else:
            v_trend = 0
            
        # Size-based capacitor selection
        if self.emergency_mode:
            # Emergency: Use large capacitors first
            return self._emergency_capacitor_control(v_min, v_max)
            
        elif v_avg < 0.98 or (v_avg < 0.99 and v_trend < -0.005):
            # Need voltage support
            return self._voltage_support_control(voltages, v_avg, v_trend, problem_buses)
            
        elif v_avg > 1.02 or (v_avg > 1.01 and v_trend > 0.005):
            # Overvoltage - turn off all capacitors
            return np.zeros(self.num_caps)
            
        else:
            # Normal operation - optimize for efficiency
            return self._efficiency_optimization_control(voltages, v_avg)
            
    def _emergency_capacitor_control(self, v_min, v_max):
        """Emergency control using large capacitors."""
        cap_actions = np.zeros(self.num_caps)
        
        if v_min < 0.95:
            # Turn on large capacitors first
            for i in self.large_caps:
                if self.time_since_switch[i] >= self.min_switch_intervals[i]:
                    cap_actions[i] = 0.7 * self.cap_ratings[i] / 10  # Convert to p.u.
                    
            # If still need more, add medium caps
            if v_min < 0.93:
                for i in self.medium_caps:
                    if self.time_since_switch[i] >= self.min_switch_intervals[i]:
                        cap_actions[i] = 0.7 * self.cap_ratings[i] / 10
                        
        return cap_actions
        
    def _voltage_support_control(self, voltages, v_avg, v_trend, problem_buses):
        """Intelligent voltage support based on location and size."""
        cap_actions = np.zeros(self.num_caps)
        
        # Calculate required var support
        var_deficit = (1.0 - v_avg) * 50  # Rough estimate in MVAr
        
        # First, check if problem is localized
        if problem_buses and len(problem_buses) < 5:
            # Localized issue - use nearby capacitors
            for bus in problem_buses:
                # Find closest capacitor
                distances = [abs(bus - cap_bus) for cap_bus in self.cap_buses]
                sorted_caps = sorted(range(self.num_caps), key=lambda i: distances[i])
                
                for cap_idx in sorted_caps[:2]:  # Use 2 closest
                    if (self.time_since_switch[cap_idx] >= self.min_switch_intervals[cap_idx] and
                        cap_actions[cap_idx] == 0):
                        # Scale by size - large caps provide more
                        if cap_idx in self.large_caps:
                            cap_actions[cap_idx] = 0.7 * self.cap_ratings[cap_idx] / 10
                        elif cap_idx in self.medium_caps:
                            cap_actions[cap_idx] = 0.6 * self.cap_ratings[cap_idx] / 10
                        else:
                            cap_actions[cap_idx] = 0.5 * self.cap_ratings[cap_idx] / 10
                            
        else:
            # Distributed issue - optimize total vars
            remaining_deficit = var_deficit
            
            # Sort capacitors by efficiency (size/switching_cost ratio)
            efficiency = self.cap_ratings / self.switching_costs
            sorted_by_efficiency = sorted(range(self.num_caps), 
                                        key=lambda i: efficiency[i], 
                                        reverse=True)
            
            for cap_idx in sorted_by_efficiency:
                if remaining_deficit <= 0:
                    break
                    
                if self.time_since_switch[cap_idx] >= self.min_switch_intervals[cap_idx]:
                    # Determine optimal utilization
                    if cap_idx in self.large_caps and remaining_deficit > 2.0:
                        cap_actions[cap_idx] = 0.7 * self.cap_ratings[cap_idx] / 10
                        remaining_deficit -= 0.7 * self.cap_ratings[cap_idx]
                    elif cap_idx in self.medium_caps and remaining_deficit > 0.8:
                        cap_actions[cap_idx] = 0.6 * self.cap_ratings[cap_idx] / 10
                        remaining_deficit -= 0.6 * self.cap_ratings[cap_idx]
                    elif cap_idx in self.small_caps and remaining_deficit > 0.2:
                        cap_actions[cap_idx] = 0.5 * self.cap_ratings[cap_idx] / 10
                        remaining_deficit -= 0.5 * self.cap_ratings[cap_idx]
                        
        return cap_actions
        
    def _efficiency_optimization_control(self, voltages, v_avg):
        """Fine-tune voltage using small capacitors for efficiency."""
        cap_actions = np.zeros(self.num_caps)
        
        # Calculate fine adjustment needed
        v_error = self.v_ref - v_avg
        
        if abs(v_error) > self.v_deadband:
            # Use small/tiny capacitors for fine control
            if v_error > 0:  # Need small boost
                # Try tiny caps first for minimal intervention
                for i in self.tiny_caps:
                    if self.time_since_switch[i] >= 10:  # Less strict timing
                        cap_actions[i] = 0.5 * self.cap_ratings[i] / 10
                        break
                        
                # If not enough, add a small cap
                if v_error > 0.01:
                    for i in self.small_caps:
                        if self.time_since_switch[i] >= self.min_switch_intervals[i]:
                            cap_actions[i] = 0.4 * self.cap_ratings[i] / 10
                            break
                            
        # Update switch history
        for i in range(self.num_caps):
            if cap_actions[i] > 0 and self.cap_switch_history[i] == 0:
                self.time_since_switch[i] = 0
                self.cap_switch_history[i] = 1
            elif cap_actions[i] == 0 and self.cap_switch_history[i] > 0:
                self.time_since_switch[i] = 0
                self.cap_switch_history[i] = 0
                
        return cap_actions
        
    def _compute_oltc_action(self, v_min, v_max, v_avg):
        """OLTC control with predictive adjustment."""
        # Only adjust if capacitors alone insufficient
        if self.emergency_mode and v_min < 0.93:
            self.oltc_position = min(1.1, self.oltc_position + 0.025)
        elif v_max > 1.07:
            self.oltc_position = max(0.9, self.oltc_position - 0.025)
        elif not self.emergency_mode:
            # Slow return to nominal
            if self.oltc_position > 1.0:
                self.oltc_position = max(1.0, self.oltc_position - 0.0125)
            elif self.oltc_position < 1.0:
                self.oltc_position = min(1.0, self.oltc_position + 0.0125)
                
        return self.oltc_position