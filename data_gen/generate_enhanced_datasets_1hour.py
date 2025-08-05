"""
Enhanced dataset generation optimized for ~1 hour runtime with detailed progress logging
- All variants included for diversity
- Clear progress indicators and ETA
- Periodic summaries
- Graceful handling of interruptions
"""
from __future__ import annotations
import argparse
import yaml
import numpy as np
from multiprocessing import Pool, Manager
from functools import partial
from pathlib import Path
from tqdm import tqdm
import sys
import time
from datetime import datetime, timedelta
import signal
import logging
from collections import defaultdict

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_env.enhanced_environment import EnhancedIEEE33Environment
from enhanced_ctrl.controller_factory import build_controllers
from data_gen.dataset_io import RLDatasetWriter
from create_algorithmic_diversity import (
    L0_Random, L1_BangBang, L2_Proportional, 
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.warning("Interrupt received! Finishing current tasks and saving...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)


def _rollout_with_progress(args):
    """Generate trajectories with progress reporting"""
    variant_name, env_cfg, episodes, seed, base_controller, max_steps, task_id, total_tasks = args
    
    start_time = time.time()
    
    try:
        # Create environment
        rng = np.random.default_rng(seed)
        env = EnhancedIEEE33Environment(
            base_load_scale=env_cfg["base_load"],
            params=env_cfg.get("params", {}),
            seed=seed,
        )
        
        # Create controller
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
        
        # Generate episodes
        for ep in range(episodes):
            if shutdown_requested:
                logger.info(f"Task {task_id}: Stopping early due to shutdown request")
                break
                
            reset_result = env.reset()
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
        
        elapsed = time.time() - start_time
        
        # Return results with metadata
        return {
            'task_id': task_id,
            'variant': variant_name,
            'env_load': env_cfg['base_load'],
            'transitions': tuples,
            'elapsed_time': elapsed,
            'episodes_completed': len(tuples) // max_steps if tuples else 0
        }
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {variant_name} in env {env_cfg['base_load']}: {e}")
        return {
            'task_id': task_id,
            'variant': variant_name,
            'env_load': env_cfg['base_load'],
            'transitions': [],
            'elapsed_time': time.time() - start_time,
            'episodes_completed': 0,
            'error': str(e)
        }


class ProgressTracker:
    """Track and report progress with ETA calculations"""
    
    def __init__(self, total_tasks):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.start_time = time.time()
        self.task_times = []
        self.controller_stats = defaultdict(lambda: {'count': 0, 'time': 0, 'transitions': 0})
        
    def update(self, result):
        self.completed_tasks += 1
        self.task_times.append(result['elapsed_time'])
        
        # Update controller statistics
        ctrl_type = result['variant'].split('_')[0]
        self.controller_stats[ctrl_type]['count'] += 1
        self.controller_stats[ctrl_type]['time'] += result['elapsed_time']
        self.controller_stats[ctrl_type]['transitions'] += len(result['transitions'])
        
        # Calculate progress and ETA
        progress = self.completed_tasks / self.total_tasks
        elapsed_total = time.time() - self.start_time
        avg_time_per_task = elapsed_total / self.completed_tasks
        remaining_tasks = self.total_tasks - self.completed_tasks
        eta_seconds = remaining_tasks * avg_time_per_task
        eta = datetime.now() + timedelta(seconds=eta_seconds)
        
        # Log progress
        logger.info(
            f"Progress: {self.completed_tasks}/{self.total_tasks} ({progress*100:.1f}%) | "
            f"Task: {result['variant'][:30]} @ {result['env_load']} | "
            f"Time: {result['elapsed_time']:.1f}s | "
            f"Transitions: {len(result['transitions'])} | "
            f"ETA: {eta.strftime('%H:%M:%S')}"
        )
        
        # Periodic summary every 10 tasks
        if self.completed_tasks % 10 == 0:
            self.print_summary()
    
    def print_summary(self):
        """Print detailed progress summary"""
        elapsed = time.time() - self.start_time
        
        logger.info("="*70)
        logger.info(f"PROGRESS SUMMARY - {datetime.now().strftime('%H:%M:%S')}")
        logger.info(f"Completed: {self.completed_tasks}/{self.total_tasks} tasks")
        logger.info(f"Elapsed time: {elapsed/60:.1f} minutes")
        logger.info(f"Average time per task: {np.mean(self.task_times):.1f}s")
        
        logger.info("\nController performance:")
        for ctrl, stats in sorted(self.controller_stats.items()):
            avg_time = stats['time'] / stats['count'] if stats['count'] > 0 else 0
            logger.info(
                f"  {ctrl}: {stats['count']} tasks, "
                f"{stats['transitions']:,} transitions, "
                f"{avg_time:.1f}s avg"
            )
        
        # Estimate total runtime
        if self.completed_tasks > 0:
            total_estimated = elapsed * self.total_tasks / self.completed_tasks
            logger.info(f"\nEstimated total runtime: {total_estimated/60:.1f} minutes")
        
        logger.info("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("datasets/ieee33_diverse_v1.h5"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-hours", type=float, default=1.0, help="Target runtime in hours")
    args = p.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("ENHANCED DATASET GENERATION - 1 HOUR VERSION")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {args.out}")
    print(f"Target runtime: {args.target_hours} hours")
    print(f"Workers: {args.n_workers}")
    print(f"Episodes per controller: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    print("="*70)
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load environment configurations
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    if env_grid_path.exists():
        logger.info(f"Loading environment configurations from {env_grid_path}")
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        logger.info("Using default environment configurations")
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.2, "params": {"noise_std": 0.1}},
            {"base_load": 1.5, "params": {"noise_std": 0.05, "spike_prob": 0.1}},
            {"base_load": 1.8, "params": {"noise_std": 0.15}},
        ]
    
    logger.info(f"Environment configurations: {len(env_grid)} variants")
    for i, cfg in enumerate(env_grid):
        logger.info(f"  Env {i+1}: load={cfg['base_load']}, params={list(cfg.get('params', {}).keys())}")
    
    # Load controllers
    base_controllers = {
        "L0_Random": L0_Random,
        "L1_BangBang": L1_BangBang,
        "L2_Proportional": L2_Proportional,
        "L3_PI_Controller": L3_PI_Controller,
        "L4_RuleBasedExpert": L4_RuleBasedExpert,
        "L5_ScipyOptimal": L5_ScipyOptimal,
    }
    
    # Load factory variants
    factory_variants = {}
    try:
        factory_variants = build_controllers()
        logger.info(f"\nController summary:")
        logger.info(f"  Base controllers: {len(base_controllers)}")
        logger.info(f"  Factory variants: {len(factory_variants)}")
        logger.info(f"  Total unique controllers: {len(base_controllers) + len(factory_variants)}")
    except Exception as e:
        logger.warning(f"Could not load factory controllers: {e}")
    
    # Prepare tasks with IDs
    tasks = []
    task_id = 0
    
    for env_cfg in env_grid:
        # Base controllers
        for ctrl_name, ctrl_class in base_controllers.items():
            seed = args.seed + abs(hash((ctrl_name, str(env_cfg)))) % 2**32
            tasks.append((
                ctrl_name, env_cfg, args.episodes, seed, ctrl_class, 
                args.max_steps, task_id, len(env_grid) * (len(base_controllers) + len(factory_variants))
            ))
            task_id += 1
        
        # Factory variants
        for v_name in factory_variants.keys():
            seed = args.seed + abs(hash((v_name, str(env_cfg)))) % 2**32
            tasks.append((
                v_name, env_cfg, args.episodes, seed, None, 
                args.max_steps, task_id, len(env_grid) * (len(base_controllers) + len(factory_variants))
            ))
            task_id += 1
    
    # Calculate expected runtime
    total_tasks = len(tasks)
    total_transitions = total_tasks * args.episodes * args.max_steps
    estimated_time = total_transitions * 0.02 / args.n_workers  # ~0.02s per transition
    
    logger.info(f"\nDataset size estimation:")
    logger.info(f"  Total tasks: {total_tasks}")
    logger.info(f"  Total transitions: {total_transitions:,}")
    logger.info(f"  Estimated runtime: {estimated_time/60:.1f} minutes")
    logger.info(f"  Transitions per task: {args.episodes * args.max_steps}")
    
    # Adjust parameters if needed to meet target runtime
    if estimated_time > args.target_hours * 3600 * 1.5:  # More than 50% over target
        logger.warning(f"Estimated runtime ({estimated_time/3600:.1f}h) exceeds target ({args.target_hours}h)")
        logger.warning("Consider reducing episodes or max_steps")
    
    print("\n" + "="*70)
    print("STARTING PARALLEL GENERATION")
    print("="*70)
    
    # Initialize progress tracker
    tracker = ProgressTracker(total_tasks)
    
    # Process tasks in parallel
    writer = None
    results = []
    start_time = time.time()
    
    try:
        with Pool(args.n_workers) as pool:
            # Submit all tasks
            async_results = [pool.apply_async(_rollout_with_progress, (task,)) for task in tasks]
            
            # Collect results with progress updates
            for async_result in async_results:
                if shutdown_requested:
                    logger.warning("Shutdown requested, saving partial results...")
                    break
                
                result = async_result.get()
                results.append(result)
                tracker.update(result)
                
    except KeyboardInterrupt:
        logger.warning("Interrupted! Saving partial results...")
    
    # Final summary
    print("\n" + "="*70)
    print("GENERATION COMPLETE - WRITING DATASET")
    print("="*70)
    
    tracker.print_summary()
    
    # Write to HDF5
    logger.info(f"\nWriting dataset to {args.out}...")
    transitions_written = 0
    failed_tasks = 0
    
    # Count total transitions
    total_transitions_actual = sum(len(r['transitions']) for r in results)
    logger.info(f"Total transitions to write: {total_transitions_actual:,}")
    
    with tqdm(total=total_transitions_actual, desc="Writing transitions") as pbar:
        for result in results:
            if 'error' in result:
                failed_tasks += 1
                continue
                
            transitions = result['transitions']
            if transitions and writer is None:
                obs_dim = len(transitions[0][0])
                act_dim = len(transitions[0][1])
                writer = RLDatasetWriter(
                    args.out, obs_dim, act_dim,
                    est_n_transitions=total_transitions_actual
                )
            
            for (obs, action, reward, n_obs, done, beh_params) in transitions:
                if writer is not None:
                    beh_id = writer.add_behaviour(beh_params["variant"], beh_params)
                    writer.append_transition(
                        obs, action, reward, n_obs, done, beh_id
                    )
                    transitions_written += 1
                    pbar.update(1)
    
    # Close and report
    if writer:
        writer.close()
        
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Dataset saved to: {args.out}")
    print(f"Total transitions: {transitions_written:,}")
    if writer:
        print(f"Total behaviors: {len(writer.beh_idx)}")
    print(f"Failed tasks: {failed_tasks}")
    print(f"Success rate: {(len(results)-failed_tasks)/len(results)*100:.1f}%")
    print(f"Average speed: {transitions_written/total_time:.1f} transitions/second")
    print("="*70)


if __name__ == "__main__":
    main()