"""
Reusable controller decorators for measurement noise, delayed actuation,
biased random sampling, hysteresis, etc.
"""
from __future__ import annotations
import numpy as np
from typing import Protocol
from functools import wraps
import inspect


class Controller(Protocol):
    """Structural typing for any object with `act(env) -> np.ndarray`."""
    def act(self, env) -> np.ndarray: ...


# ------------------------------------------------------------------ #
# Decorator factories
# ------------------------------------------------------------------ #
def noisy_measurement(sigma: float):
    """Inject Gaussian noise into voltage magnitudes before controller logic."""
    def decorator(ctrl_cls):
        orig_act = ctrl_cls.act
        @wraps(orig_act)
        def act(self, env):
            sim = env.unwrapped.simulator
            v = np.array([np.abs(b.v) for b in sim.buses.values()])
            v_noisy = v + self._rng.normal(0.0, sigma, v.shape)
            # Temporarily replace the voltages in the simulator
            original_v = [b.v for b in sim.buses.values()]
            for i, bus in enumerate(sim.buses.values()):
                # Keep phase, only change magnitude
                phase = np.angle(bus.v)
                bus.v = v_noisy[i] * np.exp(1j * phase)
            try:
                result = orig_act(self, env)
            finally:
                # Restore original voltages
                for i, bus in enumerate(sim.buses.values()):
                    bus.v = original_v[i]
            return result
        ctrl_cls.act = act
        return ctrl_cls
    return decorator


def action_delay(k_steps: int):
    """Apply FIFO delay to controller output."""
    def decorator(ctrl_cls):
        orig_init = ctrl_cls.__init__
        orig_act = ctrl_cls.act
        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            self._fifo = []
            self._default_action = None
        def act(self, env):
            a = orig_act(self, env)
            
            # Initialize default action on first call
            if self._default_action is None:
                self._default_action = np.zeros_like(a)
                # Set all controls to safe defaults within bounds
                # For renewable P (indices 0-4): set to minimum (20% of capacity)
                self._default_action[0:3] = 0.01  # 20% of 0.05
                self._default_action[3:5] = 0.02  # 20% of 0.10
                # For renewable Q (indices 5-9): set to 0 (no reactive support)
                self._default_action[5:10] = 0.0
                # For capacitors (indices 10-15): all OFF
                self._default_action[10:16] = 0.0
                # For OLTC (index 16): set to nominal
                self._default_action[16] = 1.0
            
            self._fifo.append(a)
            if len(self._fifo) <= k_steps:
                # Return the properly initialized default action
                return self._default_action.copy()
            return self._fifo.pop(0)
        ctrl_cls.__init__ = __init__
        ctrl_cls.act = act
        return ctrl_cls
    return decorator