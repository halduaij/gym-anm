"""
CLI to generate the full diversity dataset defined by YAML parameter grids.
WITH DETAILED PROGRESS MONITORING FOR DEBUGGING

Usage
-----
python -m data_gen.generate_enhanced_datasets_debug \
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
import time
import traceback
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_env.enhanced_environment import EnhancedIEEE33Environment
from enhanced_ctrl.controller_factory import build_controllers
from data_gen.dataset_io import RLDatasetWriter
from create_algorithmic_diversity import (
    L0_Random, L1_BangBang, L2_Proportional, 
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)


def _rollout(variant_name, env_cfg, episodes, seed, base_controller=None):
    """Generate (obs, action, reward, next_obs, done, beh_id) tuples with detailed progress."""
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting rollout for {variant_name}")
    print(f"  Environment config: base_load={env_cfg['base_load']}, params={env_cfg.get('params', {})}")
    print(f"  Episodes: {episodes}, Seed: {seed}")
    
    try:
        # Initialize environment
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating environment...")
        rng = np.random.default_rng(seed)
        env = EnhancedIEEE33Environment(
            base_load_scale=env_cfg["base_load"],
            params=env_cfg.get("params", {}),
            seed=seed,
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Environment created successfully")
        
        # Initialize controller
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Creating controller...")
        if base_controller is not None:
            controller = base_controller(env)
            print(f"  Using base controller: {base_controller.__name__}")
        else:
            ctrl_factory = build_controllers()
            if variant_name in ctrl_factory:
                controller = ctrl_factory[variant_name](env)
                print(f"  Using factory controller: {variant_name}")
            else:
                raise ValueError(f"Unknown controller variant: {variant_name}")
        
        beh_params = {"variant": variant_name, "env_cfg": env_cfg}
        tuples = []
        
        # Episode loop with progress tracking
        for ep in range(episodes):
            ep_start = time.time()
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Episode {ep+1}/{episodes}")
            
            # Reset environment
            print(f"  Resetting environment...")
            reset_start = time.time()
            reset_result = env.reset()
            # Handle both tuple and non-tuple returns
            if isinstance(reset_result, tuple):
                obs, info = reset_result
                print(f"  Environment returned tuple (obs, info)")
            else:
                obs = reset_result
                info = {}
            reset_time = time.time() - reset_start
            print(f"  Environment reset in {reset_time:.3f}s")
            print(f"  Initial observation shape: {obs.shape if hasattr(obs, 'shape') else len(obs)}")
            
            done = False
            step_count = 0
            ep_rewards = []
            
            # Step loop with timeout
            step_start_time = time.time()
            max_steps_per_episode = 1000  # Add step limit
            while not done and step_count < max_steps_per_episode:
                step_count += 1
                
                # Timeout check - if episode takes too long
                if time.time() - step_start_time > 300:  # 5 minute timeout per episode
                    print(f"  WARNING: Episode timeout after {step_count} steps!")
                    break
                
                # Log progress every 100 steps
                if step_count % 100 == 0:
                    elapsed = time.time() - step_start_time
                    print(f"  Step {step_count}: {elapsed:.1f}s elapsed, avg reward: {np.mean(ep_rewards) if ep_rewards else 0:.3f}")
                
                # Controller action
                try:
                    action_start = time.time()
                    action = controller.act(env)
                    action_time = time.time() - action_start
                    
                    if step_count == 1 or step_count % 500 == 0:
                        print(f"    Controller action computed in {action_time:.3f}s")
                        print(f"    Action shape: {action.shape if hasattr(action, 'shape') else len(action)}")
                except Exception as e:
                    print(f"  ERROR in controller.act(): {e}")
                    traceback.print_exc()
                    break
                
                # Environment step
                try:
                    step_start = time.time()
                    n_obs, reward, done, truncated, info = env.step(action)
                    step_time = time.time() - step_start
                    
                    if step_count == 1 or step_count % 500 == 0:
                        print(f"    Environment step computed in {step_time:.3f}s")
                        print(f"    Reward: {reward:.3f}, Done: {done}")
                    
                    ep_rewards.append(reward)
                except Exception as e:
                    print(f"  ERROR in env.step(): {e}")
                    traceback.print_exc()
                    break
                
                # Store transition
                tuples.append((obs, action, reward, n_obs, done, beh_params))
                obs = n_obs
                
                # Check if environment terminated
                if done or truncated:
                    done = True
            
            # Episode summary
            ep_time = time.time() - ep_start
            print(f"  Episode completed: {step_count} steps in {ep_time:.1f}s")
            print(f"  Total reward: {sum(ep_rewards):.3f}, Avg reward: {np.mean(ep_rewards):.3f}")
            print(f"  Transitions collected so far: {len(tuples)}")
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Rollout completed for {variant_name}")
        print(f"  Total transitions: {len(tuples)}")
        return tuples
        
    except Exception as e:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ERROR in rollout for {variant_name}: {e}")
        traceback.print_exc()
        return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("datasets/ieee33_diverse_v1.h5"))
    p.add_argument("--episodes", type=int, default=10)  # Reduced for testing
    p.add_argument("--n-workers", type=int, default=1)  # Serial for debugging
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Dataset Generation Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Output file: {args.out}")
    print(f"Episodes per controller: {args.episodes}")
    print(f"Workers: {args.n_workers}")
    print(f"Seed: {args.seed}")
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load or create default configs
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    ctrl_grid_path = Path(__file__).parent / "configs/ctrl_grid.yaml"
    
    if env_grid_path.exists():
        print(f"\nLoading environment grid from {env_grid_path}")
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        print("\nUsing default environment configurations")
        # Default environment configurations
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.5, "params": {"noise_std": 0.1}},
            {"base_load": 1.8, "params": {"noise_std": 0.05, "spike_prob": 0.1}},
        ]
    
    print(f"Environment configurations: {len(env_grid)} variants")
    for i, cfg in enumerate(env_grid):
        print(f"  {i+1}. base_load={cfg['base_load']}, params={cfg.get('params', {})}")
    
    # Use base controllers from create_algorithmic_diversity.py
    base_controllers = {
        "L0_Random": L0_Random,
        "L1_BangBang": L1_BangBang,
        "L2_Proportional": L2_Proportional,
        "L3_PI_Controller": L3_PI_Controller,
        "L4_RuleBasedExpert": L4_RuleBasedExpert,
        "L5_ScipyOptimal": L5_ScipyOptimal,
    }
    
    print(f"\nBase controllers: {list(base_controllers.keys())}")
    
    # Try to load controller variants from factory
    try:
        factory_variants = build_controllers()
        print(f"Factory controllers: {list(factory_variants.keys())}")
    except Exception as e:
        print(f"Warning: Could not load factory controllers: {e}")
        factory_variants = {}
    
    writer = None
    total_start = time.time()
    
    if args.n_workers == 1:
        # Serial execution for debugging
        results = []
        task_count = 0
        total_tasks = len(env_grid) * (len(base_controllers) + len(factory_variants))
        
        for env_idx, env_cfg in enumerate(env_grid):
            print(f"\n{'='*60}")
            print(f"Environment {env_idx+1}/{len(env_grid)}: base_load={env_cfg['base_load']}")
            print(f"{'='*60}")
            
            # First use base controllers
            for ctrl_name, ctrl_class in base_controllers.items():
                task_count += 1
                print(f"\n[Task {task_count}/{total_tasks}] {ctrl_name} in env {env_cfg['base_load']}")
                
                start_time = time.time()
                seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
                transitions = _rollout(ctrl_name, env_cfg, args.episodes, seed, ctrl_class)
                elapsed = time.time() - start_time
                
                print(f"\n[Task {task_count}/{total_tasks}] Completed in {elapsed:.1f} seconds")
                print(f"  Generated {len(transitions)} transitions")
                
                if transitions:
                    results.append(transitions)
                else:
                    print(f"  WARNING: No transitions generated!")
            
            # Then add factory variants if available
            for v_name in factory_variants.keys():
                task_count += 1
                print(f"\n[Task {task_count}/{total_tasks}] {v_name} in env {env_cfg['base_load']}")
                
                start_time = time.time()
                seed = args.seed + hash((v_name, str(env_cfg)))
                transitions = _rollout(v_name, env_cfg, args.episodes, seed)
                elapsed = time.time() - start_time
                
                print(f"\n[Task {task_count}/{total_tasks}] Completed in {elapsed:.1f} seconds")
                print(f"  Generated {len(transitions)} transitions")
                
                if transitions:
                    results.append(transitions)
    else:
        # Parallel execution
        print(f"\nRunning in parallel with {args.n_workers} workers")
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
    print(f"\n{'='*60}")
    print("Writing dataset to HDF5...")
    print(f"{'='*60}")
    
    transitions_written = 0
    for i, transitions in enumerate(results):
        if transitions:
            print(f"Processing result {i+1}/{len(results)}: {len(transitions)} transitions")
            
            if writer is None:
                obs_dim = len(transitions[0][0])
                act_dim = len(transitions[0][1])
                print(f"Initializing writer: obs_dim={obs_dim}, act_dim={act_dim}")
                writer = RLDatasetWriter(
                    args.out, obs_dim, act_dim,
                    est_n_transitions=len(transitions) * len(results)
                )
            
            for (obs, action, reward, n_obs, done, beh_params) in transitions:
                beh_id = writer.add_behaviour(beh_params["variant"], beh_params)
                writer.append_transition(
                    obs, action, reward, n_obs, done, beh_id
                )
                transitions_written += 1
                
                if transitions_written % 1000 == 0:
                    print(f"  Written {transitions_written} transitions...")
    
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
    else:
        print("\nERROR: No data generated!")


if __name__ == "__main__":
    main()