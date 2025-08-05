"""
Fast version of dataset generation with reasonable defaults
- Shorter episodes (200 steps instead of 1000)
- Fewer controller variants (base controllers only)
- Progress bar for better monitoring
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
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_env.enhanced_environment import EnhancedIEEE33Environment
from enhanced_ctrl.controller_factory import build_controllers
from data_gen.dataset_io import RLDatasetWriter
from create_algorithmic_diversity import (
    L0_Random, L1_BangBang, L2_Proportional, 
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)


def _rollout(variant_name, env_cfg, episodes, seed, base_controller=None, max_steps=200):
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
        reset_result = env.reset()
        # Handle both tuple and non-tuple returns
        if isinstance(reset_result, tuple):
            obs, info = reset_result
        else:
            obs = reset_result
        done = False
        step_count = 0
        
        while not done and step_count < max_steps:
            step_count += 1
            action = controller.act(env)
            n_obs, reward, done, truncated, info = env.step(action)
            tuples.append((obs, action, reward, n_obs, done, beh_params))
            obs = n_obs
            
    return tuples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("datasets/ieee33_diverse_v1.h5"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200, help="Max steps per episode")
    p.add_argument("--n-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-only", action="store_true", help="Use only base controllers (no variants)")
    p.add_argument("--quick", action="store_true", help="Quick mode: 1 env config, base controllers only")
    args = p.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Fast Dataset Generation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Output file: {args.out}")
    print(f"Episodes per controller: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Workers: {args.n_workers}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create default configs
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    
    if args.quick:
        # Quick mode: just one simple environment
        print("\nQuick mode: Using single environment config")
        env_grid = [{"base_load": 1.0, "params": {"noise_std": 0.05}}]
    elif env_grid_path.exists():
        print(f"\nLoading environment grid from {env_grid_path}")
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        print("\nUsing default environment configurations")
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.5, "params": {"noise_std": 0.1}},
        ]
    
    print(f"Environment configurations: {len(env_grid)} variants")
    
    # Use base controllers
    base_controllers = {
        "L0_Random": L0_Random,
        "L1_BangBang": L1_BangBang,
        "L2_Proportional": L2_Proportional,
        "L3_PI_Controller": L3_PI_Controller,
        "L4_RuleBasedExpert": L4_RuleBasedExpert,
        "L5_ScipyOptimal": L5_ScipyOptimal,
    }
    
    print(f"Base controllers: {list(base_controllers.keys())}")
    
    # Load factory variants if not in base-only mode
    factory_variants = {}
    if not args.base_only and not args.quick:
        try:
            factory_variants = build_controllers()
            print(f"Factory controllers: {len(factory_variants)} variants")
        except Exception as e:
            print(f"Warning: Could not load factory controllers: {e}")
    
    # Calculate total tasks
    total_controllers = len(base_controllers) + len(factory_variants)
    total_tasks = len(env_grid) * total_controllers
    print(f"\nTotal configurations: {total_tasks}")
    print(f"Estimated data points: {total_tasks * args.episodes * args.max_steps}")
    
    writer = None
    total_start = time.time()
    
    # Prepare all tasks
    tasks = []
    for env_cfg in env_grid:
        # Base controllers
        for ctrl_name, ctrl_class in base_controllers.items():
            seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
            tasks.append((ctrl_name, env_cfg, args.episodes, seed, ctrl_class, args.max_steps))
        
        # Factory variants
        for v_name in factory_variants.keys():
            seed = args.seed + abs(hash((v_name, str(env_cfg)))) % 2**32
            tasks.append((v_name, env_cfg, args.episodes, seed, None, args.max_steps))
    
    print(f"\nGenerating data for {len(tasks)} configurations...")
    
    if args.n_workers == 1:
        # Serial execution with progress bar
        results = []
        for task in tqdm(tasks, desc="Generating trajectories"):
            result = _rollout(*task)
            results.append(result)
    else:
        # Parallel execution
        with Pool(args.n_workers) as pool:
            futures = []
            for task in tasks:
                futures.append(pool.apply_async(_rollout, task))
            
            # Collect results with progress bar
            results = []
            for future in tqdm(futures, desc="Generating trajectories"):
                results.append(future.get())
    
    # Write to HDF5
    print(f"\nWriting dataset to {args.out}...")
    transitions_written = 0
    
    with tqdm(total=sum(len(r) for r in results), desc="Writing transitions") as pbar:
        for i, transitions in enumerate(results):
            if transitions:
                if writer is None:
                    obs_dim = len(transitions[0][0])
                    act_dim = len(transitions[0][1])
                    total_transitions = sum(len(r) for r in results)
                    writer = RLDatasetWriter(
                        args.out, obs_dim, act_dim,
                        est_n_transitions=total_transitions
                    )
                
                for (obs, action, reward, n_obs, done, beh_params) in transitions:
                    beh_id = writer.add_behaviour(beh_params["variant"], beh_params)
                    writer.append_transition(
                        obs, action, reward, n_obs, done, beh_id
                    )
                    transitions_written += 1
                    pbar.update(1)
    
    if writer:
        writer.close()
        total_time = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"Dataset Generation Completed!")
        print(f"{'='*60}")
        print(f"Dataset saved to: {args.out}")
        print(f"Total transitions: {writer._size}")
        print(f"Total behaviors: {len(writer.beh_idx)}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Transitions per second: {transitions_written/total_time:.1f}")
    else:
        print("\nERROR: No data generated!")


if __name__ == "__main__":
    main()