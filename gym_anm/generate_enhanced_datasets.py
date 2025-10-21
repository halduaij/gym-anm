"""
CLI to generate the full diversity dataset defined by YAML parameter grids.

Usage
-----
python -m data_gen.generate_enhanced_datasets \
    --out datasets/ieee33_diverse_v1.h5 \
    --episodes 100 \
    --n-workers 8 \
    --seed 42
"""
from __future__ import annotations
import argparse
import yaml
import numpy as np
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from enhanced_env.enhanced_environment import EnhancedIEEE33Environment
from enhanced_ctrl.controller_factory import build_controllers
from data_gen.dataset_io import RLDatasetWriter
from create_algorithmic_diversity import (
    L0_Random, L1_BangBang, L2_Proportional, 
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)


def _rollout(variant_name, env_cfg, episodes, seed, base_controller=None):
    """Generate (obs, action, reward, next_obs, done, beh_id) tuples."""
    rng = np.random.default_rng(seed)
    env = EnhancedIEEE33Environment(
        base_load_scale=env_cfg["base_load"],
        params=env_cfg.get("params", {}),
        seed=seed,
    )
    
    # Use base controller if provided, otherwise use factory
    if base_controller is not None:
        controller = base_controller(env)
    else:
        ctrl_factory = build_controllers()
        if variant_name in ctrl_factory:
            controller = ctrl_factory[variant_name](env)
        else:
            raise ValueError(f"Unknown controller variant: {variant_name}")
    
    beh_params = {"variant": variant_name, "env_cfg": env_cfg}
    tuples = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        while not done:
            action = controller.act(env)
            n_obs, reward, done, truncated, info = env.step(action)
            tuples.append((obs, action, reward, n_obs, done, beh_params))
            obs = n_obs
    return tuples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("datasets/ieee33_diverse_v1.h5"))
    p.add_argument("--episodes", type=int, default=10)  # Reduced for testing
    p.add_argument("--n-workers", type=int, default=1)  # Serial for debugging
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create default configs
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    ctrl_grid_path = Path(__file__).parent / "configs/ctrl_grid.yaml"
    
    if env_grid_path.exists():
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        # Default environment configurations
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.5, "params": {"noise_std": 0.1}},
            {"base_load": 1.8, "params": {"noise_std": 0.05, "spike_prob": 0.1}},
        ]
    
    # Use base controllers from create_algorithmic_diversity.py
    base_controllers = {
        "L0_Random": L0_Random,
        "L1_BangBang": L1_BangBang,
        "L2_Proportional": L2_Proportional,
        "L3_PI_Controller": L3_PI_Controller,
        "L4_RuleBasedExpert": L4_RuleBasedExpert,
        "L5_ScipyOptimal": L5_ScipyOptimal,
    }
    
    # Try to load controller variants from factory
    factory_variants = build_controllers()
    
    writer = None
    
    if args.n_workers == 1:
        # Serial execution for debugging
        results = []
        for env_cfg in env_grid:
            # First use base controllers
            for ctrl_name, ctrl_class in base_controllers.items():
                print(f"Generating data for {ctrl_name} in env {env_cfg['base_load']}...")
                import time
                start_time = time.time()
                seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
                transitions = _rollout(ctrl_name, env_cfg, args.episodes, seed, ctrl_class)
                elapsed = time.time() - start_time
                print(f"  Completed in {elapsed:.1f} seconds. Generated {len(transitions)} transitions.")
                results.append(transitions)
            
            # Then add factory variants if available
            for v_name in factory_variants.keys():
                print(f"Generating data for {v_name} in env {env_cfg['base_load']}...")
                seed = args.seed + hash((v_name, str(env_cfg)))
                transitions = _rollout(v_name, env_cfg, args.episodes, seed)
                results.append(transitions)
    else:
        # Parallel execution
        with Pool(args.n_workers) as pool:
            results = []
            for env_cfg in env_grid:
                for ctrl_name, ctrl_class in base_controllers.items():
                    seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
                    results.append(
                        pool.apply_async(
                            _rollout,
                            (ctrl_name, env_cfg, args.episodes, seed, ctrl_class),
                        )
                    )
                for v_name in factory_variants.keys():
                    seed = args.seed + hash((v_name, str(env_cfg)))
                    results.append(
                        pool.apply_async(
                            _rollout,
                            (v_name, env_cfg, args.episodes, seed),
                        )
                    )
            
            # Collect results
            results = [res.get() for res in tqdm(results)]
    
    # Write to HDF5
    for i, transitions in enumerate(results):
        if writer is None and transitions:
            obs_dim = len(transitions[0][0])
            act_dim = len(transitions[0][1])
            writer = RLDatasetWriter(
                args.out, obs_dim, act_dim,
                est_n_transitions=len(transitions) * len(results)
            )
        for (obs, action, reward, n_obs, done, beh_params) in transitions:
            beh_id = writer.add_behaviour(beh_params["variant"], beh_params)
            writer.append_transition(
                obs, action, reward, n_obs, done, beh_id
            )
    
    if writer:
        writer.close()
        print(f"\nDataset saved to {args.out}")
        print(f"Total transitions: {writer._size}")
        print(f"Total behaviors: {len(writer.beh_idx)}")
    else:
        print("No data generated!")


if __name__ == "__main__":
    main()