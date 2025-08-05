"""
Practical dataset generation with ALL variants but optimized for speed
- Keeps all 39 controller variants for diversity
- Shorter episodes (300 steps) for faster generation
- Optimized parallel processing
- Better progress monitoring
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


def _rollout(args):
    """Generate (obs, action, reward, next_obs, done, beh_id) tuples."""
    variant_name, env_cfg, episodes, seed, base_controller, max_steps = args
    
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
    p.add_argument("--max-steps", type=int, default=300, help="Max steps per episode")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--env-subset", type=int, default=None, help="Use only first N environment configs")
    args = p.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Practical Dataset Generation (All Variants)")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Output file: {args.out}")
    print(f"Episodes per controller: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print(f"Workers: {args.n_workers}")
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create default configs
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    
    if env_grid_path.exists():
        print(f"\nLoading environment grid from {env_grid_path}")
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        print("\nUsing default environment configurations")
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.2, "params": {"noise_std": 0.1}},
            {"base_load": 1.5, "params": {"noise_std": 0.05, "spike_prob": 0.1}},
            {"base_load": 1.8, "params": {"noise_std": 0.15}},
        ]
    
    # Option to use subset of environments
    if args.env_subset:
        env_grid = env_grid[:args.env_subset]
        print(f"Using subset of {len(env_grid)} environment configs")
    
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
    
    # Load factory variants - WE WANT THESE FOR DIVERSITY
    factory_variants = {}
    try:
        factory_variants = build_controllers()
        print(f"\nControllers:")
        print(f"  Base controllers: {len(base_controllers)} ({list(base_controllers.keys())})")
        print(f"  Factory variants: {len(factory_variants)}")
        print(f"  Total controllers: {len(base_controllers) + len(factory_variants)}")
    except Exception as e:
        print(f"Warning: Could not load factory controllers: {e}")
    
    # Calculate total tasks
    total_controllers = len(base_controllers) + len(factory_variants)
    total_tasks = len(env_grid) * total_controllers
    total_transitions = total_tasks * args.episodes * args.max_steps
    
    print(f"\nDataset size estimation:")
    print(f"  Total configurations: {total_tasks}")
    print(f"  Estimated transitions: {total_transitions:,}")
    print(f"  Estimated time: {total_tasks * args.episodes * args.max_steps * 0.02 / args.n_workers / 60:.1f} minutes")
    
    writer = None
    total_start = time.time()
    
    # Prepare all tasks
    tasks = []
    for env_cfg in env_grid:
        # Base controllers
        for ctrl_name, ctrl_class in base_controllers.items():
            seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
            tasks.append((ctrl_name, env_cfg, args.episodes, seed, ctrl_class, args.max_steps))
        
        # Factory variants - INCLUDE ALL OF THEM
        for v_name in factory_variants.keys():
            seed = args.seed + abs(hash((v_name, str(env_cfg)))) % 2**32
            tasks.append((v_name, env_cfg, args.episodes, seed, None, args.max_steps))
    
    print(f"\nGenerating data for {len(tasks)} configurations...")
    print(f"Using {args.n_workers} parallel workers")
    
    if args.n_workers == 1:
        # Serial execution with progress bar
        results = []
        for task in tqdm(tasks, desc="Generating trajectories"):
            result = _rollout(task)
            results.append(result)
    else:
        # Parallel execution with better batching
        print("\nStarting parallel generation...")
        with Pool(args.n_workers) as pool:
            # Use imap_unordered for better performance
            results = list(tqdm(
                pool.imap_unordered(_rollout, tasks, chunksize=2),
                total=len(tasks),
                desc="Generating trajectories"
            ))
    
    # Write to HDF5
    print(f"\nWriting dataset to {args.out}...")
    transitions_written = 0
    
    # Calculate total transitions for progress bar
    total_transitions_actual = sum(len(r) for r in results)
    print(f"Total transitions to write: {total_transitions_actual:,}")
    
    with tqdm(total=total_transitions_actual, desc="Writing transitions") as pbar:
        for i, transitions in enumerate(results):
            if transitions:
                if writer is None:
                    obs_dim = len(transitions[0][0])
                    act_dim = len(transitions[0][1])
                    writer = RLDatasetWriter(
                        args.out, obs_dim, act_dim,
                        est_n_transitions=total_transitions_actual
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
        print(f"Total transitions: {writer._size:,}")
        print(f"Total behaviors: {len(writer.beh_idx)}")
        print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"Transitions per second: {transitions_written/total_time:.1f}")
        
        # Show breakdown by controller type
        print(f"\nBehavior breakdown:")
        behavior_counts = {}
        for beh_name in writer.beh_idx.keys():
            base_name = beh_name.split('_')[0]
            behavior_counts[base_name] = behavior_counts.get(base_name, 0) + 1
        for name, count in sorted(behavior_counts.items()):
            print(f"  {name}: {count} variants")
    else:
        print("\nERROR: No data generated!")


if __name__ == "__main__":
    main()