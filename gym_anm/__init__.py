"""A package for designing RL ANM tasks in power grids."""

from gymnasium.envs.registration import register

from .agents import MPCAgentPerfect, MPCAgentConstant
from .envs import ANMEnv
from .envs.ieee33_env import IEEE33Env

register(
    id="ANM6Easy-v0",
    entry_point="gym_anm.envs:ANM6Easy",
)

register(
    id="IEEE33-v0",
    entry_point="gym_anm.envs.ieee33_env:IEEE33Env",
)
