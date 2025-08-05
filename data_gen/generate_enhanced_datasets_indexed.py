"""
Final enhanced dataset generation with indexed discrete actions:
1. Proper episode termination at load profile boundaries
2. Unique behaviors per controller-environment combination  
3. Capacitor actions stored as binary indices (0/1)
4. OLTC stored as tap index (0-15)

This format is much easier for RL algorithms to learn!
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
import h5py
import json

sys.path.append(str(Path(__file__).parent.parent))

from enhanced_env.enhanced_environment import EnhancedIEEE33Environment
from enhanced_ctrl.controller_factory import build_controllers
from create_algorithmic_diversity import (
    L0_Random, L1_BangBang, L2_Proportional, 
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)
from action_index_mapping import ActionIndexMapper

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


def discretize_oltc(oltc_value, n_taps=16):
    """Discretize continuous OLTC value to nearest tap position."""
    tap_positions = np.linspace(0.9, 1.1, n_taps)
    oltc_value = np.clip(oltc_value, 0.9, 1.1)
    idx = np.argmin(np.abs(tap_positions - oltc_value))
    return tap_positions[idx]


class IndexedRLDatasetWriter:
    """Dataset writer that stores actions with discrete indices."""
    
    def __init__(self, path: str | Path, obs_dim: int, act_dim: int,
                 est_n_transitions: int, compress: bool = True):
        self.file = h5py.File(path, "w")
        c = "gzip" if compress else None
        
        # Create datasets
        self.dset = {
            "obs": self.file.create_dataset(
                "transitions/obs", (0, obs_dim), maxshape=(None, obs_dim),
                dtype="f4", compression=c
            ),
            "action": self.file.create_dataset(
                "transitions/action", (0, act_dim), maxshape=(None, act_dim),
                dtype="f4", compression=c
            ),
            "action_continuous": self.file.create_dataset(
                "transitions/action_continuous", (0, act_dim), maxshape=(None, act_dim),
                dtype="f4", compression=c
            ),
            "reward": self.file.create_dataset(
                "transitions/reward", (0,), maxshape=(None,), dtype="f4", compression=c
            ),
            "next_obs": self.file.create_dataset(
                "transitions/next_obs", (0, obs_dim), maxshape=(None, obs_dim),
                dtype="f4", compression=c
            ),
            "done": self.file.create_dataset(
                "transitions/done", (0,), maxshape=(None,), dtype="u1", compression=c
            ),
            "beh_id": self.file.create_dataset(
                "transitions/beh_id", (0,), maxshape=(None,), dtype="i4", compression=c
            ),
        }
        self._size = 0
        self.beh_table_grp = self.file.create_group("behaviour")
        self.beh_idx = {}
        
        # Store action space info
        self.action_mapper = ActionIndexMapper()
        action_info = self.action_mapper.get_action_info()
        
        # Create metadata group
        meta_grp = self.file.create_group("metadata")
        meta_grp.attrs["action_space_type"] = "mixed_indexed"
        meta_grp.attrs["renewable_indices"] = action_info["renewable_p"]["indices"]
        meta_grp.attrs["capacitor_indices"] = action_info["capacitors"]["indices"]
        meta_grp.attrs["oltc_index"] = action_info["oltc"]["index"]
        meta_grp.attrs["oltc_tap_positions"] = action_info["oltc"]["tap_positions"]
        meta_grp.attrs["capacitor_ratings"] = action_info["capacitors"]["ratings_mvar"]
        
    def _grow(self, n: int):
        for ds in self.dset.values():
            ds.resize(self._size + n, axis=0)
    
    def add_behaviour(self, variant: str, beh_params: dict) -> int:
        """Create unique behavior ID including full configuration."""
        # Create unique key from controller AND environment
        env_cfg = beh_params.get("env_cfg", {})
        base_load = env_cfg.get("base_load", 1.0)
        
        # Include other important env params if they vary
        noise_std = env_cfg.get("params", {}).get("noise_std", 0.0)
        
        # Create fully unique identifier
        unique_key = f"{variant}_load{base_load}_noise{noise_std}"
        
        if unique_key in self.beh_idx:
            return self.beh_idx[unique_key]
            
        idx = len(self.beh_idx)
        self.beh_idx[unique_key] = idx
        
        # Store in HDF5
        id_ds = self.beh_table_grp.require_dataset(
            "id", (idx + 1,), dtype="i4", maxshape=(None,), exact=False
        )
        if id_ds.shape[0] <= idx:
            id_ds.resize((idx + 1,))
        id_ds[idx] = idx
        
        name_ds = self.beh_table_grp.require_dataset(
            "name", (idx + 1,), dtype=h5py.string_dtype(), maxshape=(None,), exact=False
        )
        if name_ds.shape[0] <= idx:
            name_ds.resize((idx + 1,))
        name_ds[idx] = unique_key
        
        params_ds = self.beh_table_grp.require_dataset(
            "params", (idx + 1,), dtype=h5py.string_dtype(), maxshape=(None,), exact=False
        )
        if params_ds.shape[0] <= idx:
            params_ds.resize((idx + 1,))
        params_ds[idx] = json.dumps(beh_params)
        
        return idx
    
    def append_transition(self, obs, action_continuous, reward, n_obs, done, beh_id):
        """Append single transition with both continuous and indexed actions."""
        self._grow(1)
        idx = self._size
        
        # Convert continuous action to indices
        action_indexed = self.action_mapper.continuous_to_indices(action_continuous)
        
        self.dset["obs"][idx] = obs
        self.dset["action"][idx] = action_indexed  # Store indexed version
        self.dset["action_continuous"][idx] = action_continuous  # Also store original
        self.dset["reward"][idx] = reward
        self.dset["next_obs"][idx] = n_obs
        self.dset["done"][idx] = done
        self.dset["beh_id"][idx] = beh_id
        
        self._size += 1
        
    def close(self):
        self.file.attrs["total_transitions"] = self._size
        self.file.attrs["num_behaviors"] = len(self.beh_idx)
        self.file.attrs["num_episodes"] = int(np.sum(self.dset["done"][:]))
        
        # Add description
        self.file.attrs["description"] = (
            "Actions stored with discrete indices:\n"
            "- Renewable P/Q (0-9): Continuous values\n" 
            "- Capacitors (10-15): Binary indices 0/1 (OFF/ON)\n"
            "- OLTC (16): Tap index 0-15 (maps to 0.9-1.1)\n"
            "Use 'action' for indexed version, 'action_continuous' for original values"
        )
        
        self.file.close()


def create_daily_load_profile(base_scale=1.0):
    """Create a 24-hour load profile function."""
    def profile(t):
        """Standard workday load profile."""
        hour = (t % 96) / 4  # Convert to hours (96 steps = 24 hours)
        
        if hour < 6:  # Night
            return base_scale * (0.7 + 0.05 * np.sin(hour * np.pi / 6))
        elif hour < 9:  # Morning ramp
            return base_scale * (0.7 + 0.3 * (hour - 6) / 3)
        elif hour < 17:  # Workday
            return base_scale * (1.0 + 0.1 * np.sin((hour - 9) * np.pi / 4))
        elif hour < 20:  # Evening peak
            return base_scale * (1.2 - 0.1 * np.sin((hour - 17) * np.pi / 3))
        else:  # Night ramp down
            return base_scale * (1.0 - 0.3 * (hour - 20) / 4)
            
    return profile


def _rollout_with_load_profile(args):
    """Generate trajectories with load profile-based episodes"""
    variant_name, env_cfg, episodes, seed, base_controller, profile_steps, task_id, total_tasks = args
    
    start_time = time.time()
    
    try:
        # Create environment
        rng = np.random.default_rng(seed)
        env = EnhancedIEEE33Environment(
            base_load_scale=1.0,  # Will be scaled by profile
            params=env_cfg.get("params", {}),
            seed=seed,
        )
        
        # Create load profile
        load_profile = create_daily_load_profile(env_cfg["base_load"])
        
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
        
        # Generate episodes based on load profile cycles
        for ep in range(episodes):
            if shutdown_requested:
                logger.info(f"Task {task_id}: Stopping early due to shutdown request")
                break
                
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                obs, info = reset_result
            else:
                obs = reset_result
            
            # Run for one complete load profile cycle
            for step in range(profile_steps):
                # Update load from profile
                env.load_scale = load_profile(step)
                
                # Get action and step
                action = controller.act(env)
                
                # CRITICAL: Discretize OLTC to 16 tap positions
                action[16] = discretize_oltc(action[16], n_taps=16)
                
                # Validate action before stepping
                if action[16] < 0.9 or action[16] > 1.1:
                    logger.warning(f"Task {task_id}: Invalid OLTC value {action[16]}, clipping to valid range")
                    action[16] = np.clip(action[16], 0.9, 1.1)
                
                n_obs, reward, env_done, truncated, info = env.step(action)
                
                # Episode ends at profile boundary OR network collapse
                done = (step == profile_steps - 1) or env_done
                
                tuples.append((obs, action, reward, n_obs, done, beh_params))
                obs = n_obs
                
                if env_done:  # Network collapsed
                    break
        
        elapsed = time.time() - start_time
        
        # Return results with metadata
        return {
            'task_id': task_id,
            'variant': variant_name,
            'env_load': env_cfg['base_load'],
            'transitions': tuples,
            'elapsed_time': elapsed,
            'episodes_completed': ep + 1 if tuples else 0,
            'profile_steps': profile_steps
        }
        
    except Exception as e:
        logger.error(f"Task {task_id} failed: {variant_name} in env {env_cfg['base_load']}: {e}")
        import traceback
        logger.error(traceback.format_exc())
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
        self.failed_tasks = []
        
    def update(self, result):
        self.completed_tasks += 1
        self.task_times.append(result['elapsed_time'])
        
        # Track failures
        if 'error' in result:
            self.failed_tasks.append({
                'task_id': result['task_id'],
                'variant': result['variant'],
                'env_load': result['env_load'],
                'error': result['error']
            })
        
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
        logger.info(f"Failed: {len(self.failed_tasks)} tasks")
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
        
        if self.failed_tasks:
            logger.info(f"\nFailed tasks: {len(self.failed_tasks)}")
            for failure in self.failed_tasks[-5:]:  # Show last 5
                logger.info(f"  - {failure['variant']} @ load {failure['env_load']}: {failure['error'][:50]}...")
        
        # Estimate total runtime
        if self.completed_tasks > 0:
            total_estimated = elapsed * self.total_tasks / self.completed_tasks
            logger.info(f"\nEstimated total runtime: {total_estimated/60:.1f} minutes")
        
        logger.info("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("datasets/ieee33_indexed.h5"))
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--profile-steps", type=int, default=96, help="Steps per load profile (96=24hrs)")
    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-hours", type=float, default=1.0, help="Target runtime in hours")
    args = p.parse_args()
    
    # Header
    print("\n" + "="*70)
    print("INDEXED DATASET GENERATION - OPTIMIZED FOR RL")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output file: {args.out}")
    print(f"Target runtime: {args.target_hours} hours")
    print(f"Workers: {args.n_workers}")
    print(f"Episodes per controller: {args.episodes}")
    print(f"Steps per episode (load profile): {args.profile_steps}")
    print("\nACTION INDEXING:")
    print("✅ Renewable P/Q (0-9): Continuous values")
    print("✅ Capacitors (10-15): Binary indices 0/1 (OFF/ON)")
    print("✅ OLTC (16): Discrete index 0-15 (16 tap positions)")
    print("\nALL FIXES APPLIED:")
    print("✅ Episodes terminate at load profile boundaries")
    print("✅ Unique behavior IDs per controller-environment")
    print("✅ Actions stored as discrete indices for easy learning")
    print("="*70)
    
    # Create output directory
    args.out.parent.mkdir(parents=True, exist_ok=True)
    
    # Show action mapping info
    mapper = ActionIndexMapper()
    print("\nOLTC Tap Mapping:")
    tap_positions = mapper.oltc_taps
    for i in range(0, 16, 4):
        print(f"  Index {i:2d}-{min(i+3,15):2d}: {tap_positions[i]:.3f} - {tap_positions[min(i+3,15)]:.3f}")
    
    # Load environment configurations
    env_grid_path = Path(__file__).parent / "configs/env_grid.yaml"
    if env_grid_path.exists():
        logger.info(f"\nLoading environment configurations from {env_grid_path}")
        env_grid = yaml.safe_load(env_grid_path.read_text())
    else:
        logger.info("\nUsing default environment configurations")
        env_grid = [
            {"base_load": 1.0, "params": {"noise_std": 0.05}},
            {"base_load": 1.2, "params": {"noise_std": 0.1}},
            {"base_load": 1.5, "params": {"noise_std": 0.05, "spike_prob": 0.1}},
            {"base_load": 1.8, "params": {"noise_std": 0.15}},
        ]
    
    logger.info(f"Environment configurations: {len(env_grid)} variants")
    for i, cfg in enumerate(env_grid):
        logger.info(f"  Env {i+1}: load={cfg['base_load']}, params={list(cfg.get('params', {}).keys())}")
    
    # Load base controllers
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
                args.profile_steps, task_id, len(env_grid) * (len(base_controllers) + len(factory_variants))
            ))
            task_id += 1
        
        # Factory variants
        for v_name in factory_variants.keys():
            seed = args.seed + abs(hash((v_name, str(env_cfg)))) % 2**32
            tasks.append((
                v_name, env_cfg, args.episodes, seed, None, 
                args.profile_steps, task_id, len(env_grid) * (len(base_controllers) + len(factory_variants))
            ))
            task_id += 1
    
    # Calculate expected runtime
    total_tasks = len(tasks)
    total_transitions = total_tasks * args.episodes * args.profile_steps
    estimated_time = total_transitions * 0.02 / args.n_workers  # ~0.02s per transition
    
    logger.info(f"\nDataset size estimation:")
    logger.info(f"  Total tasks: {total_tasks}")
    logger.info(f"  Total episodes: {total_tasks * args.episodes}")
    logger.info(f"  Total transitions: {total_transitions:,}")
    logger.info(f"  Estimated runtime: {estimated_time/60:.1f} minutes")
    logger.info(f"  Transitions per episode: {args.profile_steps}")
    
    # Adjust parameters if needed to meet target runtime
    if estimated_time > args.target_hours * 3600 * 1.5:  # More than 50% over target
        logger.warning(f"Estimated runtime ({estimated_time/3600:.1f}h) exceeds target ({args.target_hours}h)")
        logger.warning("Consider reducing episodes or using only base controllers")
    
    print("\n" + "="*70)
    print("STARTING PARALLEL GENERATION WITH INDEXED ACTIONS")
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
            async_results = [pool.apply_async(_rollout_with_load_profile, (task,)) for task in tasks]
            
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
    print("GENERATION COMPLETE - WRITING INDEXED DATASET")
    print("="*70)
    
    tracker.print_summary()
    
    # Write to HDF5 with indexed writer
    logger.info(f"\nWriting dataset to {args.out}...")
    transitions_written = 0
    failed_tasks = 0
    unique_behaviors = set()
    oltc_indices = []
    cap_indices = {i: [] for i in range(10, 16)}
    
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
                writer = IndexedRLDatasetWriter(
                    args.out, obs_dim, act_dim,
                    est_n_transitions=total_transitions_actual
                )
            
            for (obs, action, reward, n_obs, done, beh_params) in transitions:
                if writer is not None:
                    beh_id = writer.add_behaviour(beh_params["variant"], beh_params)
                    
                    # Track unique behaviors
                    env_cfg = beh_params["env_cfg"]
                    base_load = env_cfg.get("base_load", 1.0)
                    noise_std = env_cfg.get("params", {}).get("noise_std", 0.0)
                    unique_key = f"{beh_params['variant']}_load{base_load}_noise{noise_std}"
                    unique_behaviors.add(unique_key)
                    
                    # Write transition (writer will convert to indices)
                    writer.append_transition(
                        obs, action, reward, n_obs, done, beh_id
                    )
                    
                    # Track discrete values for verification
                    action_indexed = writer.action_mapper.continuous_to_indices(action)
                    oltc_indices.append(int(action_indexed[16]))
                    for i in range(10, 16):
                        cap_indices[i].append(int(action_indexed[i]))
                    
                    transitions_written += 1
                    pbar.update(1)
    
    # Close and report
    if writer:
        writer.close()
        
    total_time = time.time() - start_time
    
    # Verify discretization
    unique_oltc_indices = np.unique(oltc_indices)
    all_oltc_valid = all(0 <= idx <= 15 for idx in unique_oltc_indices)
    
    print("\n" + "="*70)
    print("FINAL SUMMARY - INDEXED DATASET")
    print("="*70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total runtime: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"Dataset saved to: {args.out}")
    print(f"Total transitions: {transitions_written:,}")
    print(f"Total unique behaviors: {len(unique_behaviors)}")
    print(f"Total episodes: {sum(1 for r in results if 'error' not in r for t in r['transitions'] if t[4])}")
    print(f"Failed tasks: {failed_tasks}")
    print(f"Success rate: {(len(results)-failed_tasks)/len(results)*100:.1f}%")
    print(f"Average speed: {transitions_written/total_time:.1f} transitions/second")
    
    # Action indexing verification
    print(f"\nAction Indexing Verification:")
    print(f"OLTC indices used: {sorted(unique_oltc_indices)} (expected: 0-15)")
    print(f"OLTC indexing valid: {all_oltc_valid}")
    
    print("\nCapacitor indices (should be 0 or 1):")
    for i in range(10, 16):
        unique_cap = np.unique(cap_indices[i])
        valid = all(idx in [0, 1] for idx in unique_cap)
        print(f"  Capacitor {i-10} (action[{i}]): {unique_cap} - Valid: {valid}")
    
    # List unique behaviors
    print("\nUnique behaviors recorded:")
    for i, beh in enumerate(sorted(unique_behaviors)[:10]):  # Show first 10
        print(f"  {i+1}. {beh}")
    if len(unique_behaviors) > 10:
        print(f"  ... and {len(unique_behaviors) - 10} more")
    
    # Show failed tasks details
    if tracker.failed_tasks:
        print(f"\nFailed tasks ({len(tracker.failed_tasks)} total):")
        for failure in tracker.failed_tasks[:10]:  # Show first 10
            print(f"  - Task {failure['task_id']}: {failure['variant']} @ load {failure['env_load']}")
            print(f"    Error: {failure['error']}")
    
    print("="*70)
    print("✅ INDEXED DATASET GENERATION COMPLETE:")
    print("   - Episodes terminate at load profile boundaries")
    print("   - Each controller-environment combo has unique behavior ID")
    print("   - Capacitor actions stored as binary indices 0/1")
    print("   - OLTC stored as tap index 0-15")
    print("   - Ready for efficient RL training!")
    print("="*70)


if __name__ == "__main__":
    main()