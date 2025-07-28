"""IEEE33 environment with renewable generators and all fixes."""

import numpy as np
from gym_anm.envs.ieee33_env import IEEE33Env
from gym_anm.simulator.components.devices import RenewableGen, Load
from copy import deepcopy


def create_renewable_network():
    """Create IEEE33 network with renewable generators."""
    # Import the base network
    from .network import network as base_network
    
    # Deep copy to avoid modifying the original
    network = deepcopy(base_network)
    
    # The base network already has capacitors and OLTC, just add renewables
    # Keep as numpy array to preserve 2D structure
    devices = network["device"]
    
    # Find the next available device ID
    if len(devices) > 0:
        next_dev_id = int(max([dev[0] for dev in devices]) + 1)
    else:
        next_dev_id = 36  # Start renewables at 36
    
    # Convert to list to append new devices
    device_list = devices.tolist()
    
    # Solar generators (0.5 MW each at buses 5, 11, 29)
    solar_buses = [5, 11, 29]
    solar_power = 0.5  # MW
    
    for bus in solar_buses:
        p_max = solar_power / network["baseMVA"]
        q_max = p_max * 0.4
        
        device = [
            next_dev_id,      # DEV_ID
            bus,              # BUS_ID
            2,                # DEV_TYPE (RenewableGen)
            None,             # Q/P ratio (not used for generators)
            p_max,            # PMAX
            0.0,              # PMIN
            q_max,            # QMAX
            -q_max,           # QMIN
            p_max * 0.7,      # P+
            None,             # P-
            q_max * 0.8,      # Q+
            -q_max * 0.8,     # Q-
            None,             # SOC_MAX
            None,             # SOC_MIN
            None              # EFF
        ]
        device_list.append(device)
        next_dev_id += 1
    
    # Wind generators (1.0 MW each at buses 14, 30)
    wind_buses = [14, 30]
    wind_power = 1.0  # MW
    
    for bus in wind_buses:
        p_max = wind_power / network["baseMVA"]
        q_max = p_max * 0.4
        
        device = [
            next_dev_id,      # DEV_ID
            bus,              # BUS_ID
            2,                # DEV_TYPE (RenewableGen)
            None,             # Q/P ratio (not used for generators)
            p_max,            # PMAX
            0.0,              # PMIN
            q_max,            # QMAX
            -q_max,           # QMIN
            p_max * 0.7,      # P+
            None,             # P-
            q_max * 0.8,      # Q+
            -q_max * 0.8,     # Q-
            None,             # SOC_MAX
            None,             # SOC_MIN
            None              # EFF
        ]
        device_list.append(device)
        next_dev_id += 1
    
    # Convert back to numpy array
    network["device"] = np.array(device_list, dtype=object)
    return network


class IEEE33RenewableEnv(IEEE33Env):
    """
    IEEE33 with renewable generators and all fixes.
    
    Key features:
    1. 5 renewable generators (3 solar, 2 wind)
    2. Proper load scaling (Expert 2's solution)
    3. Time-varying renewable potential
    4. Branch rate fix required after reset
    """
    
    def __init__(self, load_scale=1.0, scenario='default', **kwargs):
        # Store parameters
        self.load_scale = load_scale
        self.scenario = scenario
        
        # First initialize parent
        super().__init__()
        
        # Then reinitialize with renewable network
        network = create_renewable_network()
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
        self._load_scale_override = None  # For dynamic load testing
        
        # Calculate total nominal load
        self._load_ids = [dev_id for dev_id, dev in self.simulator.devices.items()
                          if isinstance(dev, Load)]
        self.total_nominal_load = sum(abs(self.simulator.devices[dev_id].p_min) 
                                      for dev_id in self._load_ids) * self.simulator.baseMVA
        
        print(f"IEEE33RenewableEnv initialized:")
        print(f"  Base MVA: {self.simulator.baseMVA}")
        print(f"  Total nominal load: {self.total_nominal_load:.2f} MW")
        print(f"  Load scale: {self.load_scale}")
        print(f"  Scaled load: {self.total_nominal_load * self.load_scale:.2f} MW")
        print(f"  Note: Branch rates need to be set after reset!")
    
    def init_state(self):
        """Initialize state."""
        n_dev = self.simulator.N_device
        n_des = self.simulator.N_des
        n_gen = self.simulator.N_non_slack_gen
        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)
        
        # Small random initialization for stability
        state += np.random.normal(0, 0.001, size=state.shape)
        
        return state
    
    def reset(self, seed=None, options=None):
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        self.timestep = 0
        self.hour_of_day = np.random.uniform(0, 24)
        
        # Call parent reset
        obs, info = super().reset(seed=seed, options=options)
        
        # Fix branch rates (critical!)
        self._fix_branch_rates()
        
        # Update renewable potential
        self._update_renewable_potential()
        
        return obs, info
    
    def step(self, action):
        """Step with time progression."""
        self.timestep += 1
        self.hour_of_day = (self.hour_of_day + self.delta_t / 3600) % 24
        
        # Update renewable potential
        self._update_renewable_potential()
        
        # Call parent step
        obs, reward, terminated, truncated, info = super().step(action)
        
        return obs, reward, terminated, truncated, info
    
    def next_vars(self, s_t):
        """
        Return MW values for loads - simulator will convert to p.u.
        This implements Expert 2's critical fix.
        """
        n_vars = self.simulator.N_load + self.simulator.N_non_slack_gen + self.K
        vars = np.zeros(n_vars)
        
        # Time-based variations
        hour = self.hour_of_day
        time_factor = 0.8 + 0.3 * np.sin((hour - 3) * np.pi / 12)
        
        # Use override if set (for testing), otherwise use configured load_scale
        effective_load_scale = self._load_scale_override if self._load_scale_override is not None else self.load_scale
        
        # Set load values in MW (negative for consumption)
        for idx, dev_id in enumerate(self._load_ids):
            if idx < self.simulator.N_load:
                dev = self.simulator.devices[dev_id]
                # p_min is in p.u., convert to MW
                nominal_mw = abs(dev.p_min) * self.simulator.baseMVA
                # Add noise
                noise = 1.0 + np.random.normal(0, 0.02)
                # Return as NEGATIVE MW for loads
                vars[idx] = -nominal_mw * effective_load_scale * time_factor * noise
        
        return vars
    
    def _update_renewable_potential(self):
        """Update renewable generation potential based on time."""
        hour = self.hour_of_day
        
        # Solar profile (peak at noon)
        if 6 <= hour <= 18:
            solar_factor = np.sin((hour - 6) * np.pi / 12)
        else:
            solar_factor = 0
        
        # Wind profile (higher at night/morning)
        wind_factor = 0.6 + 0.4 * np.cos((hour - 6) * np.pi / 12)
        
        # Scenario variations
        if self.scenario == 'high_renewable':
            solar_factor *= 1.2
            wind_factor *= 1.2
        elif self.scenario == 'low_renewable':
            solar_factor *= 0.5
            wind_factor *= 0.5
        
        # Update potentials
        for dev_id, device in self.simulator.devices.items():
            if isinstance(device, RenewableGen):
                if dev_id in [36, 37, 38]:  # Solar
                    device.p_pot = device.p_max * solar_factor
                else:  # Wind (39, 40)
                    device.p_pot = device.p_max * wind_factor
    
    def _fix_branch_rates(self):
        """
        Fix branch rate limits to reasonable values.
        This is called automatically on reset.
        
        IEEE33 has all branch rates set to 0, which causes any 
        power flow to be considered a violation. This sets realistic
        MVA limits based on typical distribution system values.
        """
        for i, (bid, branch) in enumerate(self.simulator.branches.items()):
            if i < 5:  # Main feeder
                branch.rate = 1.2  # 12 MVA
            elif i < 15:  # Primary laterals
                branch.rate = 0.5  # 5 MVA
            elif i < 25:  # Secondary laterals
                branch.rate = 0.3  # 3 MVA
            else:  # End branches
                branch.rate = 0.2  # 2 MVA