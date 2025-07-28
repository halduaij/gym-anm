"""IEEE33 environment with unequal capacitors designed to expose L2's weaknesses."""

import numpy as np
from gym_anm.simulator.components.devices import CapacitorBank
from .ieee33_renewable_complete import IEEE33RenewableEnv, create_renewable_network
from copy import deepcopy


def create_unequal_capacitor_network():
    """Create IEEE33 network with 6 capacitors of very different sizes."""
    # Start with renewable network
    network = create_renewable_network()
    
    # Remove existing capacitors (devices 8 and 9)
    device_list = network["device"].tolist()
    device_list = [dev for dev in device_list if dev[0] not in [8, 9]]
    
    # Find next device ID
    next_dev_id = int(max([dev[0] for dev in device_list]) + 1)
    
    # Add 6 capacitors with very unequal sizes
    # This creates a challenging optimization problem where simple proportional
    # control will be inefficient
    new_capacitors = [
        # Large capacitor at critical location
        {
            'bus': 17,  # Main feeder critical point
            'q_max': 0.30,  # 3.0 MVAr - LARGE
            'description': 'Primary voltage support'
        },
        # Medium capacitors at strategic locations
        {
            'bus': 24,  # Lateral 1
            'q_max': 0.15,  # 1.5 MVAr - MEDIUM
            'description': 'Lateral 1 support'
        },
        {
            'bus': 30,  # Lateral 2
            'q_max': 0.12,  # 1.2 MVAr - MEDIUM
            'description': 'Lateral 2 support'
        },
        # Small capacitors for fine control
        {
            'bus': 8,   # Early feeder
            'q_max': 0.05,  # 0.5 MVAr - SMALL
            'description': 'Fine voltage adjustment'
        },
        {
            'bus': 12,  # Mid feeder
            'q_max': 0.03,  # 0.3 MVAr - VERY SMALL
            'description': 'Minimal support'
        },
        # Tiny capacitor - almost useless alone but good for fine-tuning
        {
            'bus': 32,  # End of system
            'q_max': 0.01,  # 0.1 MVAr - TINY
            'description': 'End-of-line fine tuning'
        }
    ]
    
    for cap_info in new_capacitors:
        device = [
            next_dev_id,           # DEV_ID
            cap_info['bus'],       # BUS_ID
            4,                     # DEV_TYPE (CapacitorBank)
            None,                  # Q/P ratio
            0.0,                   # PMAX
            0.0,                   # PMIN
            cap_info['q_max'],     # QMAX (in p.u.)
            0.0,                   # QMIN
            None,                  # P+
            None,                  # P-
            cap_info['q_max'],     # Q+
            0.0,                   # Q-
            None,                  # SOC_MAX
            None,                  # SOC_MIN
            None                   # EFF
        ]
        device_list.append(device)
        next_dev_id += 1
    
    network["device"] = np.array(device_list, dtype=object)
    return network


class IEEE33UnequalCapacitorsEnv(IEEE33RenewableEnv):
    """
    IEEE33 with 6 capacitors of very different sizes.
    
    Capacitor sizes (MVAr): [3.0, 1.5, 1.2, 0.5, 0.3, 0.1]
    Total: 6.6 MVAr (vs 3.715 MW load)
    
    This configuration specifically challenges simple controllers:
    - L2 will waste the large capacitor or underutilize small ones
    - Optimal control requires intelligent coordination
    - Small capacitors are only useful for fine-tuning
    """
    
    def __init__(self, switching_cost_multiplier=1.0, **kwargs):
        # Store switching cost multiplier before calling parent init
        self.switching_cost_multiplier = switching_cost_multiplier
        
        # Call parent init WITHOUT custom network first
        super().__init__(**kwargs)
        
        # Get capacitor info
        self.capacitor_ids = []
        self.capacitor_buses = []
        self.capacitor_ratings = []
        
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, CapacitorBank):
                self.capacitor_ids.append(dev_id)
                self.capacitor_buses.append(dev.bus_id)
                self.capacitor_ratings.append(dev.q_max * self.simulator.baseMVA)
        
        # Sort by rating to ensure consistent ordering
        sorted_indices = sorted(range(len(self.capacitor_ratings)), 
                               key=lambda i: self.capacitor_ratings[i], reverse=True)
        self.capacitor_ids = [self.capacitor_ids[i] for i in sorted_indices]
        self.capacitor_buses = [self.capacitor_buses[i] for i in sorted_indices]
        self.capacitor_ratings = [self.capacitor_ratings[i] for i in sorted_indices]
        
        # Track previous capacitor states for switching cost
        self.prev_capacitor_states = np.zeros(len(self.capacitor_ids))
        self.total_switches = 0
        self.switching_costs = 0.0
        
        # Base switching costs proportional to capacitor size
        # Larger capacitors have higher switching costs
        self.base_switching_costs = [
            0.01 * rating * self.switching_cost_multiplier 
            for rating in self.capacitor_ratings
        ]
        
        print(f"IEEE33UnequalCapacitorsEnv initialized:")
        print(f"  Total capacitors: {len(self.capacitor_ids)}")
        print(f"  Capacitor locations: buses {self.capacitor_buses}")
        print(f"  Capacitor ratings (MVAr): {[f'{r:.1f}' for r in self.capacitor_ratings]}")
        print(f"  Switching costs: {[f'{c:.4f}' for c in self.base_switching_costs]}")
        print(f"  Total capacity: {sum(self.capacitor_ratings):.1f} MVAr")
        print(f"  Nominal load: {self.total_nominal_load:.2f} MW")
        
    def step(self, action):
        """Override step to track switching costs."""
        # Extract capacitor actions (indices 10-15)
        cap_actions = action[10:16]
        
        # Calculate switching costs
        switches = np.abs(cap_actions - self.prev_capacitor_states) > 0.01
        step_switching_cost = np.sum(switches * self.base_switching_costs)
        self.total_switches += np.sum(switches)
        self.switching_costs += step_switching_cost
        
        # Update previous states
        self.prev_capacitor_states = cap_actions.copy()
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        # Subtract switching cost from reward
        reward -= step_switching_cost
        
        # Add switching info
        info['switching_cost'] = step_switching_cost
        info['total_switches'] = self.total_switches
        info['cumulative_switching_cost'] = self.switching_costs
        
        return obs, reward, terminated, truncated, info
        
    def get_capacitor_info(self):
        """Return information about capacitors for controller adaptation."""
        return {
            'num_capacitors': len(self.capacitor_ids),
            'capacitor_buses': self.capacitor_buses,
            'capacitor_ratings': self.capacitor_ratings,
            'switching_costs': self.base_switching_costs
        }