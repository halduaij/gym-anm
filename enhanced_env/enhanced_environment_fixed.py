"""
Enhanced IEEE-33 Distribution-Network Environment (FIXED)
=========================================================
Adds stochastic load, renewable and measurement noise, device delays and random
fault injection, while *preserving the Gym-ANM interface*.

CRITICAL FIX: Properly implements next_vars() to provide renewable potential values
in MW, preventing the p_pot = 99+ p.u. issue.
"""
from __future__ import annotations

import numpy as np
from ready_to_use_l5_implementation import IEEE33ProperEnvironment
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
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


class EnhancedIEEE33Environment(IEEE33ProperEnvironment):
    """
    IEEE-33 environment with stochastic load & measurement perturbations.
    
    FIXED: Properly provides renewable potential values in next_vars()

    Args
    ----
    base_load_scale : float
        Nominal scaling applied to the *static* load curve of the parent env.
    params : StochasticParams | dict | None
        Parameter set controlling stochastic processes.
    seed : int | None
        RNG seed for full reproducibility.
    """
    metadata = {"render.modes": []}

    def __init__(
        self,
        base_load_scale: float = 1.0,
        params: Optional[StochasticParams | Dict[str, Any]] = None,
        seed: Optional[int] = None,
        curtailment_weight: float = 40.0,
    ):
        self.base_load_scale = base_load_scale
        self.params = (
            params if isinstance(params, StochasticParams)
            else StochasticParams(**(params or {}))
        )
        self._rng = np.random.default_rng(seed)
        super().__init__(load_scale=base_load_scale, curtailment_weight=curtailment_weight)

        # --- internal state for correlated processes
        self._load_drift = 0.0
        self._delayed_actions = []            # circular buffer
        self._current_step = 0
        self._device_mask = np.ones(
            self.action_space.shape, dtype=bool
        )  # True = available
        
        # Track time for renewable profiles
        self.hour_of_day = 12.0  # Start at noon
        
        # Identify renewable generators (assuming they are the last 5 non-slack generators)
        # In IEEE33 with renewables: devices 36-40 (3 solar, 2 wind)
        self._renewable_ids = []
        from gym_anm.simulator.components import Generator
        for dev_id, dev in self.simulator.devices.items():
            if isinstance(dev, Generator) and not dev.is_slack and dev_id >= 36:
                self._renewable_ids.append(dev_id)
        self._renewable_ids.sort()
        
        print(f"Enhanced environment initialized with {len(self._renewable_ids)} renewable generators")

    # --------------------------------------------------------------------- #
    # Gym overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
        # Random time of day
        self.hour_of_day = self._rng.uniform(0, 24)
        
        result = super().reset(**kwargs)
        # Handle both old and new gym API
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        # re-draw stochastic initial conditions
        self._load_drift = 0.0
        self._current_step = 0
        self._delayed_actions.clear()
        self._device_mask[:] = True
        if self._rng.random() < self.params.device_fail_prob:
            # randomly take out 1–2 switchable capacitors
            n_faults = self._rng.integers(1, 3)
            idx = self._rng.choice(
                np.arange(self.action_space.shape[0]), size=n_faults, replace=False
            )
            self._device_mask[idx] = False
        augmented_obs = self._augment_obs(obs)
        return augmented_obs, info

    def step(self, action: np.ndarray):
        # Update time
        self.hour_of_day = (self.hour_of_day + self.delta_t) % 24
        
        self._apply_load_fluctuations()

        # action delay model
        if self._rng.random() < self.params.action_delay_prob:
            delay = self._rng.integers(1, self.params.max_delay + 1)
        else:
            delay = 0
        self._delayed_actions.append((delay, action.copy()))

        # Determine which action is executed this step
        # Default: use nominal values for a valid action
        exec_action = action.copy()  # Start with current action as default
        if self._delayed_actions:
            # decrement all counters
            self._delayed_actions = [
                (d - 1, a) for (d, a) in self._delayed_actions
            ]
            # execute those at d == 0
            ready = [(d, a) for (d, a) in self._delayed_actions if d <= 0]
            self._delayed_actions = [
                (d, a) for (d, a) in self._delayed_actions if d > 0
            ]
            if ready:
                exec_action = ready[-1][1]  # FIFO: last ready
        # mask failed devices
        exec_action = exec_action * self._device_mask

        obs, reward, done, truncated, info = super().step(exec_action)
        obs = self._augment_obs(obs)
        self._current_step += 1
        return obs, reward, done, truncated, info
    
    def next_vars(self, s_t):
        """
        Return load values and renewable potential in MW.
        
        CRITICAL: This override fixes the p_pot = 99+ p.u. issue by properly
        providing renewable potential values.
        """
        n_vars = self.simulator.N_load + self.simulator.N_non_slack_gen + self.K
        vars = np.zeros(n_vars)
        
        # First, handle loads (from parent implementation)
        for idx, dev_id in enumerate(self._load_ids):
            if idx < self.simulator.N_load:
                dev = self.simulator.devices[dev_id]
                nominal_mw = abs(dev.p_min) * self.simulator.baseMVA
                vars[idx] = -nominal_mw * self.load_scale
        
        # Now handle renewable potential (THIS IS THE FIX)
        # After loads, we need to provide potential for ALL non-slack generators
        gen_idx = 0
        from gym_anm.simulator.components import Generator
        for dev_id, dev in sorted(self.simulator.devices.items()):
            if isinstance(dev, Generator) and not dev.is_slack:
                if gen_idx < self.simulator.N_non_slack_gen:
                    idx = self.simulator.N_load + gen_idx
                    
                    if dev_id in self._renewable_ids:
                        # This is a renewable generator - provide time-varying potential
                        nominal_mw = dev.p_max * self.simulator.baseMVA
                        
                        # Determine if solar or wind based on device ID
                        # Devices 36-38 are solar (0.5 MW each)
                        # Devices 39-40 are wind (1.0 MW each)
                        if dev_id <= 38:  # Solar
                            # Solar profile (peak at noon)
                            if 6 <= self.hour_of_day <= 18:
                                solar_factor = np.sin((self.hour_of_day - 6) * np.pi / 12)
                            else:
                                solar_factor = 0
                            # Add some noise
                            solar_factor *= (1.0 + self._rng.normal(0, 0.05))
                            solar_factor = np.clip(solar_factor, 0, 1)
                            vars[idx] = nominal_mw * solar_factor
                        else:  # Wind
                            # Wind profile (higher at night/morning)
                            wind_factor = 0.6 + 0.4 * np.cos((self.hour_of_day - 6) * np.pi / 12)
                            # Add some noise
                            wind_factor *= (1.0 + self._rng.normal(0, 0.1))
                            wind_factor = np.clip(wind_factor, 0, 1)
                            vars[idx] = nominal_mw * wind_factor
                    else:
                        # Non-renewable generator - full potential
                        vars[idx] = dev.p_max * self.simulator.baseMVA
                    
                    gen_idx += 1
        
        return vars

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _apply_load_fluctuations(self):
        """Mutate `self.load_scale` in-place according to stochastic params."""
        p = self.params
        noise_factor = 1.0 + self._rng.normal(0.0, p.noise_std)
        # sudden spike
        if self._rng.random() < p.spike_prob:
            spike = self._rng.uniform(*p.spike_range)
            noise_factor += spike

        # drift
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
        """Concatenate exogenous params so that the process stays Markov."""
        # Example: [original_obs, load_scale, meas_noise, action_delay_flag, hour_of_day]
        exogenous = np.array(
            [
                self.load_scale,
                self.params.meas_noise,
                float(self._delayed_actions != []),
                self.hour_of_day / 24.0,  # Normalized hour
            ],
            dtype=np.float32,
        )
        # add measurement noise
        noisy_obs = obs + self._rng.normal(
            0.0, self.params.meas_noise, size=obs.shape
        ).astype(obs.dtype)
        return np.concatenate([noisy_obs, exogenous], axis=0)