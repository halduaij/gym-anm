"""
Factory that auto-generates 30+ controller variants from a grid YAML file.

Each variant is wrapped **dynamically**, avoiding hundreds of boilerplate
sub-classes.
"""
from __future__ import annotations
import yaml
import itertools
import numpy as np
from types import SimpleNamespace
from pathlib import Path
from typing import Dict, List, Tuple
from .base_wrappers import noisy_measurement, action_delay


CONFIG_PATH = Path(__file__).parent.parent / "data_gen/configs/ctrl_grid.yaml"


def _wrap_ctrl(base_cls, variant_cfg):
    cls = base_cls
    if variant_cfg.get("noisy", False):
        cls = noisy_measurement(variant_cfg["sigma"])(cls)
    if d := variant_cfg.get("delay", 0):
        cls = action_delay(d)(cls)
    if "params" in variant_cfg:
        # specialisation via partial kwargs
        def _ctor(env, **kw):
            kw.update(variant_cfg["params"])
            return cls(env, **kw)
        return _ctor
    return cls


def build_controllers() -> Dict[str, callable]:
    """Return dict: variant_name -> callable(env) -> controller instance."""
    try:
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        # Return empty dict if config not found
        return {}
        
    variants = {}
    for base_name, base_cfg in cfg.items():
        # Import the controller class
        module_path, class_name = base_cfg["class_path"].rsplit(".", 1)
        try:
            import importlib
            module = importlib.import_module(module_path)
            base_cls = getattr(module, class_name)
        except (ImportError, AttributeError):
            continue
            
        grid_keys = list(base_cfg.get("grid", {}).keys())
        grid_values = list(base_cfg.get("grid", {}).values())
        for vals in itertools.product(*grid_values):
            grid_spec = dict(zip(grid_keys, vals))
            variant_cfg = base_cfg["template"] | grid_spec
            variant_name = f"{base_name}_{'_'.join(f'{k}{v}' for k,v in grid_spec.items())}"
            variants[variant_name] = lambda env, bc=base_cls, vc=variant_cfg: _wrap_ctrl(bc, vc)(env)
    return variants