"""
Enhanced IEEE-33 Distribution-Network Environment
=================================================
Adds stochastic load, renewable and measurement noise, device delays and random
fault injection, while *preserving the Gym-ANM interface*.

FIXED: Curtailment weight is now applied correctly without extreme scaling.

The class keeps the MDP Markov by *appending exogenous parameters* to the
observation; downstream agents can therefore condition on them.
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

    # --------------------------------------------------------------------- #
    # Gym overrides
    # --------------------------------------------------------------------- #
    def reset(self, **kwargs):
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
        # Example: [original_obs, load_scale, meas_noise, action_delay_flag]
        exogenous = np.array(
            [
                self.load_scale,
                self.params.meas_noise,
                float(self._delayed_actions != []),
            ],
            dtype=np.float32,
        )
        # add measurement noise
        noisy_obs = obs + self._rng.normal(
            0.0, self.params.meas_noise, size=obs.shape
        ).astype(obs.dtype)
        return np.concatenate([noisy_obs, exogenous], axis=0)