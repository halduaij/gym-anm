"""IEEE33 environment with multiple distributed capacitors for more complex control."""

import numpy as np
from gym_anm.simulator.components.devices import CapacitorBank
from .ieee33_renewable_complete import IEEE33RenewableEnv, create_renewable_network
from copy import deepcopy


def create_multi_capacitor_network():
    """Create IEEE33 network with 6 distributed capacitors instead of 2."""
    # Start with renewable network
    network = create_renewable_network()
    
    # Current network has 2 capacitors at buses 24 and 30
    # Let's add 4 more at strategic locations for voltage support
    
    device_list = network["device"].tolist()
    
    # Find next device ID (should be 41 after renewables)
    next_dev_id = int(max([dev[0] for dev in device_list]) + 1)
    
    # Additional capacitors at different voltage-critical buses
    new_capacitors = [
        # Bus 6 - Near the beginning for upstream support
        {
            'bus': 6,
            'q_max': 0.15,  # 1.5 MVAr
            'description': 'Upstream voltage support'
        },
        # Bus 12 - Middle of first lateral
        {
            'bus': 12,
            'q_max': 0.10,  # 1.0 MVAr
            'description': 'First lateral support'
        },
        # Bus 17 - End of main feeder before laterals
        {
            'bus': 17,
            'q_max': 0.20,  # 2.0 MVAr
            'description': 'Main feeder end support'
        },
        # Bus 32 - End of system
        {
            'bus': 32,
            'q_max': 0.15,  # 1.5 MVAr
            'description': 'System end support'
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


class IEEE33MultiCapacitorEnv(IEEE33RenewableEnv):
    """
    IEEE33 with 6 distributed capacitors instead of 2.
    
    This makes optimal capacitor coordination much more challenging:
    - Simple proportional control (L2) will struggle with coordination
    - L5's multi-timescale optimization should excel
    
    Action space expands from 13 to 17:
    - 0-4: Renewable P (5 generators)
    - 5-9: Renewable Q (5 generators)
    - 10-15: Capacitors (6 units) <- EXPANDED
    - 16: OLTC tap ratio
    """
    
    def __init__(self, **kwargs):
        # Skip parent init, do custom initialization
        super(IEEE33RenewableEnv, self).__init__()
        
        # Store parameters
        self.load_scale = kwargs.get('load_scale', 1.0)
        self.scenario = kwargs.get('scenario', 'default')
        
        # Create network with multiple capacitors
        network = create_multi_capacitor_network()
        from gym_anm.simulator import Simulator
        self.simulator = Simulator(network, delta_t=self.delta_t, lamb=self.lamb)
        
        # Rebuild action and observation spaces
        self.action_space = self._build_action_space()
        self.obs_values = self._build_observation_space('state')
        self.observation_space = self.observation_bounds()
        if self.observation_space is not None:
            self.observation_N = self.observation_space.shape[0]
        
        # Initialize state
        self.state = self.init_state()
        self.terminated = False
        
        # Time tracking
        self.timestep = 0
        self.hour_of_day = np.random.uniform(0, 24)
        self._load_scale_override = None
        
        # Calculate loads
        from gym_anm.simulator.components.devices import Load
        self._load_ids = [dev_id for dev_id, dev in self.simulator.devices.items()
                          if isinstance(dev, Load)]
        self.total_nominal_load = sum(abs(self.simulator.devices[dev_id].p_min) 
                                      for dev_id in self._load_ids) * self.simulator.baseMVA
        
        # Get capacitor info
        self.capacitor_ids = []
        self.capacitor_buses = []
        self.capacitor_ratings = []
        
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, CapacitorBank):
                self.capacitor_ids.append(dev_id)
                self.capacitor_buses.append(dev.bus_id)
                self.capacitor_ratings.append(dev.q_max * self.simulator.baseMVA)
        
        print(f"IEEE33MultiCapacitorEnv initialized:")
        print(f"  Total capacitors: {len(self.capacitor_ids)}")
        print(f"  Capacitor locations: buses {self.capacitor_buses}")
        print(f"  Capacitor ratings (MVAr): {self.capacitor_ratings}")
        print(f"  Action space dimension: {self.action_space.shape[0]}")
        
    def get_capacitor_info(self):
        """Return information about capacitors for controller adaptation."""
        return {
            'num_capacitors': len(self.capacitor_ids),
            'capacitor_buses': self.capacitor_buses,
            'capacitor_ratings': self.capacitor_ratings
        }