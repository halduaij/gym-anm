"""
Clean Enhanced IEEE-33 Environment without inheritance issues
"""
from __future__ import annotations

import numpy as np
from gym_anm.envs.ieee33_env.ieee33_multi_capacitor import IEEE33MultiCapacitorEnv
from gym_anm.simulator.components import RenewableGen
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import math
import random


@dataclass
class StochasticParams:
    noise_std: float = 0.05          # Gaussian load σ (fraction of base)
    spike_prob: float = 0.05         # p(load spike per step)
    spike_range: Tuple[float, float] = (0.2, 0.3)
    drift_rate: float = 0.001        # incremental drift σ
    meas_noise: float = 0.005        # σ of voltage measurement noise
    action_delay_prob: float = 0.1   # probability that delay is invoked
    max_delay: int = 3               # delay length in steps
    device_fail_prob: float = 0.05   # p(device fault at reset)


class CleanEnhancedIEEE33Environment(IEEE33MultiCapacitorEnv):
    """
    Clean enhanced IEEE-33 environment that properly handles curtailment weight.
    
    This version directly inherits from IEEE33MultiCapacitorEnv and implements
    curtailment weight correctly without the inheritance chain issues.
    """
    
    def __init__(
        self,
        base_load_scale: float = 1.0,
        params: Optional[StochasticParams | Dict[str, Any]] = None,
        seed: Optional[int] = None,
        curtailment_weight: float = 40.0,
        capacitor_switch_cost: float = 1.0,
        oltc_switch_cost: float = 5.0,
    ):
        # Initialize parent
        super().__init__()
        
        # Set parameters
        self.base_load_scale = base_load_scale
        self.load_scale = base_load_scale
        self.curtailment_weight = curtailment_weight
        self.capacitor_switch_cost = capacitor_switch_cost
        self.oltc_switch_cost = oltc_switch_cost
        self.params = (
            params if isinstance(params, StochasticParams)
            else StochasticParams(**(params or {}))
        )
        self._rng = np.random.default_rng(seed)
        
        # Internal state
        self._load_drift = 0.0
        self._delayed_actions = []
        self._current_step = 0
        self._device_mask = np.ones(self.action_space.shape, dtype=bool)
        self.hour_of_day = 12.0
        self.prev_action = None  # Track previous action for switching costs
        
        # Get device IDs
        self._load_ids = []
        self._renewable_ids = []
        
        for dev_id, dev in self.simulator.devices.items():
            if hasattr(dev, 'type') and dev.type == -1:  # Load
                self._load_ids.append(dev_id)
            elif isinstance(dev, RenewableGen):
                self._renewable_ids.append(dev_id)
                
        self._load_ids.sort()
        self._renewable_ids.sort()
        
        # Calculate total load
        total_load_mw = sum(abs(self.simulator.devices[i].p_min) * self.simulator.baseMVA 
                           for i in self._load_ids)
        
        print(f"Clean Enhanced Environment initialized:")
        print(f"  Total nominal load: {total_load_mw:.2f} MW")
        print(f"  Load scale: {base_load_scale}")
        print(f"  Curtailment weight: {curtailment_weight}")
        print(f"  Capacitor switch cost: {capacitor_switch_cost}")
        print(f"  OLTC switch cost: {oltc_switch_cost}")
        print(f"  Renewables: {len(self._renewable_ids)} devices")
        
    def reset(self, **kwargs):
        # Random time of day
        self.hour_of_day = self._rng.uniform(0, 24)
        
        # Reset parent
        obs, info = super().reset(**kwargs)
        
        # Fix branch rates
        self._fix_branch_rates()
        
        # Reset our state
        self._load_drift = 0.0
        self._current_step = 0
        self._delayed_actions.clear()
        self._device_mask[:] = True
        self.prev_action = None
        
        # Random device failures
        if self._rng.random() < self.params.device_fail_prob:
            n_faults = self._rng.integers(1, 3)
            idx = self._rng.choice(
                np.arange(self.action_space.shape[0]), size=n_faults, replace=False
            )
            self._device_mask[idx] = False
            
        # Augment observation
        augmented_obs = self._augment_obs(obs)
        return augmented_obs, info
        
    def step(self, action: np.ndarray):
        # Update time
        self.hour_of_day = (self.hour_of_day + self.delta_t) % 24
        
        # Apply load fluctuations
        self._apply_load_fluctuations()
        
        # Handle action delays
        if self._rng.random() < self.params.action_delay_prob:
            delay = self._rng.integers(1, self.params.max_delay + 1)
        else:
            delay = 0
        self._delayed_actions.append((delay, action.copy()))
        
        # Process delayed actions
        exec_action = action.copy()
        if self._delayed_actions:
            self._delayed_actions = [(d - 1, a) for (d, a) in self._delayed_actions]
            ready = [(d, a) for (d, a) in self._delayed_actions if d <= 0]
            self._delayed_actions = [(d, a) for (d, a) in self._delayed_actions if d > 0]
            if ready:
                exec_action = ready[-1][1]
                
        # Apply device mask
        exec_action = exec_action * self._device_mask
        
        # Execute step with base environment
        obs, base_reward, terminated, truncated, info = super().step(exec_action)
        
        # Apply curtailment weight adjustment
        if self.curtailment_weight != 1.0:
            # Calculate curtailment after the step
            curtailment_pu = 0.0
            for dev_id in self._renewable_ids:
                if dev_id in self.simulator.devices:
                    dev = self.simulator.devices[dev_id]
                    curtailment_pu += max(0, dev.p_pot - dev.p)
            
            # The base reward already includes curtailment at weight=1
            # We add (weight-1) * curtailment * delta_t
            additional_penalty = curtailment_pu * (self.curtailment_weight - 1.0) * self.simulator.delta_t
            base_reward -= additional_penalty
        
        # Apply switching costs
        if self.prev_action is not None:
            switching_cost = self._calculate_switching_cost(exec_action, self.prev_action)
            base_reward -= switching_cost
            info['switching_cost'] = switching_cost
            info['capacitor_switches'] = self._count_capacitor_switches(exec_action, self.prev_action)
            info['oltc_switches'] = self._count_oltc_switches(exec_action, self.prev_action)
        else:
            info['switching_cost'] = 0.0
            info['capacitor_switches'] = 0
            info['oltc_switches'] = 0
        
        # Store current action for next step
        self.prev_action = exec_action.copy()
        
        # Augment observation
        obs = self._augment_obs(obs)
        self._current_step += 1
        
        return obs, base_reward, terminated, truncated, info
        
    def next_vars(self, s_t):
        """Provide load and renewable potential values in MW."""
        n_vars = self.simulator.N_load + self.simulator.N_non_slack_gen + self.K
        vars = np.zeros(n_vars)
        
        # Loads (negative MW)
        for idx, dev_id in enumerate(self._load_ids):
            if idx < self.simulator.N_load and dev_id in self.simulator.devices:
                dev = self.simulator.devices[dev_id]
                nominal_mw = abs(dev.p_min) * self.simulator.baseMVA
                vars[idx] = -nominal_mw * self.load_scale
        
        # Renewable potential (positive MW)
        if self.simulator.N_non_slack_gen > 0:
            # Get all renewable devices in order
            renewable_devices = []
            for dev_id, dev in sorted(self.simulator.devices.items()):
                if isinstance(dev, RenewableGen):
                    renewable_devices.append((dev_id, dev))
            
            for i, (dev_id, dev) in enumerate(renewable_devices):
                if i < self.simulator.N_non_slack_gen:
                    nominal_mw = dev.p_max * self.simulator.baseMVA
                    
                    # Time-based profiles
                    if dev_id <= 38:  # Solar (devices 36-38)
                        if 6 <= self.hour_of_day <= 18:
                            solar_factor = np.sin((self.hour_of_day - 6) * np.pi / 12)
                        else:
                            solar_factor = 0
                        solar_factor *= (1.0 + self._rng.normal(0, 0.05))
                        solar_factor = np.clip(solar_factor, 0, 1)
                        vars[self.simulator.N_load + i] = nominal_mw * solar_factor
                    else:  # Wind (devices 39-40)
                        wind_factor = 0.6 + 0.4 * np.cos((self.hour_of_day - 6) * np.pi / 12)
                        wind_factor *= (1.0 + self._rng.normal(0, 0.1))
                        wind_factor = np.clip(wind_factor, 0, 1)
                        vars[self.simulator.N_load + i] = nominal_mw * wind_factor
        
        return vars
        
    def _fix_branch_rates(self):
        """Fix branch rate limits."""
        for i, (bid, branch) in enumerate(self.simulator.branches.items()):
            if i < 5:
                branch.rate = 1.2  # 12 MVA
            elif i < 15:
                branch.rate = 0.5  # 5 MVA
            elif i < 25:
                branch.rate = 0.3  # 3 MVA
            else:
                branch.rate = 0.2  # 2 MVA
                
    def _apply_load_fluctuations(self):
        """Apply stochastic load variations."""
        p = self.params
        noise_factor = 1.0 + self._rng.normal(0.0, p.noise_std)
        
        # Sudden spike
        if self._rng.random() < p.spike_prob:
            spike = self._rng.uniform(*p.spike_range)
            noise_factor += spike
            
        # Drift
        self._load_drift += self._rng.normal(0.0, p.drift_rate)
        self._load_drift = float(np.clip(self._load_drift, -0.2, 0.2))
        
        self.load_scale = float(
            np.clip(
                self.base_load_scale * noise_factor * (1.0 + self._load_drift),
                a_min=0.5,
                a_max=2.5,
            )
        )
        
    def _augment_obs(self, obs: np.ndarray) -> np.ndarray:
        """Add exogenous variables to observation."""
        exogenous = np.array(
            [
                self.load_scale,
                self.params.meas_noise,
                float(self._delayed_actions != []),
                self.hour_of_day / 24.0,
            ],
            dtype=np.float32,
        )
        
        # Add measurement noise
        noisy_obs = obs + self._rng.normal(
            0.0, self.params.meas_noise, size=obs.shape
        ).astype(obs.dtype)
        
        return np.concatenate([noisy_obs, exogenous], axis=0)
    
    def _calculate_switching_cost(self, action: np.ndarray, prev_action: np.ndarray) -> float:
        """Calculate total switching cost between actions."""
        cost = 0.0
        
        # Capacitor switching cost
        capacitor_switches = self._count_capacitor_switches(action, prev_action)
        cost += capacitor_switches * self.capacitor_switch_cost
        
        # OLTC switching cost
        oltc_switches = self._count_oltc_switches(action, prev_action)
        cost += oltc_switches * self.oltc_switch_cost
        
        return cost
        
    def _count_capacitor_switches(self, action: np.ndarray, prev_action: np.ndarray) -> int:
        """Count number of capacitor switches."""
        # Capacitor is switched if it goes from 0 to >0 or >0 to 0
        curr_caps_on = action[10:16] > 1e-6  # Small threshold for numerical stability
        prev_caps_on = prev_action[10:16] > 1e-6
        
        # Count changes in on/off state
        switches = np.sum(curr_caps_on != prev_caps_on)
        return int(switches)
        
    def _count_oltc_switches(self, action: np.ndarray, prev_action: np.ndarray) -> int:
        """Count OLTC tap changes."""
        # OLTC is considered switched if tap position changes significantly
        if abs(action[16] - prev_action[16]) > 0.01:
            return 1
        return 0