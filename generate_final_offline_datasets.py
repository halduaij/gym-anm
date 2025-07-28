"""
Final offline RL dataset generation.
Uses simple controllers to ensure it works without errors.
"""

import numpy as np
import pickle
import os
from datetime import datetime
from ready_to_use_l5_implementation import IEEE33ProperEnvironment


class SimpleL0_Random:
    """Random actions."""
    def __init__(self):
        self.name = "L0_Random"
    
    def act(self, env):
        return env.action_space.sample()


class SimpleL1_Reactive:
    """Simple reactive control."""
    def __init__(self):
        self.name = "L1_Reactive"
    
    def act(self, env):
        action = np.zeros(env.action_space.shape[0])
        # Conservative renewable usage
        action[0:3] = 0.02  # 40% of max
        action[3:5] = 0.04
        action[16] = 1.0  # No OLTC
        return action


class SimpleL2_Proportional:
    """Proportional control."""
    def __init__(self):
        self.name = "L2_Proportional"
    
    def act(self, env):
        action = np.zeros(env.action_space.shape[0])
        
        # Get voltages
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        
        # Proportional renewable
        factor = np.clip(2.0 * (1.0 - v_avg), 0.3, 0.8)
        action[0:3] = 0.05 * factor
        action[3:5] = 0.1 * factor
        
        # Simple capacitor
        if np.min(voltages) < 0.96:
            action[10] = 1.0
        
        action[16] = 1.0
        return action


class SimpleL3_Coordinated:
    """Coordinated control."""
    def __init__(self):
        self.name = "L3_Coordinated"
    
    def act(self, env):
        action = np.zeros(env.action_space.shape[0])
        
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_avg = np.mean(voltages)
        
        # Better renewable control
        if v_avg < 0.98:
            factor = 0.8
        elif v_avg > 1.02:
            factor = 0.3
        else:
            factor = 0.6
        
        action[0:3] = 0.05 * factor
        action[3:5] = 0.1 * factor
        
        # Capacitors
        if v_min < 0.95:
            action[10:12] = 1.0
        elif v_min < 0.96:
            action[10] = 1.0
        
        # Basic OLTC
        if v_avg < 0.97:
            action[16] = 0.98
        elif v_avg > 1.03:
            action[16] = 1.02
        else:
            action[16] = 1.0
        
        return action


class SimpleL4_Advanced:
    """Advanced control."""
    def __init__(self):
        self.name = "L4_Advanced"
        self.v_history = []
    
    def act(self, env):
        action = np.zeros(env.action_space.shape[0])
        
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_avg = np.mean(voltages)
        v_min = np.min(voltages)
        
        # Track history
        self.v_history.append(v_avg)
        if len(self.v_history) > 5:
            self.v_history.pop(0)
        
        # Trend-based control
        if len(self.v_history) >= 3:
            trend = self.v_history[-1] - self.v_history[-3]
        else:
            trend = 0
        
        # Predictive renewable
        if v_avg < 0.98 or trend < -0.001:
            factor = 0.85
        elif v_avg > 1.02 or trend > 0.001:
            factor = 0.2
        else:
            factor = 0.65
        
        action[0:3] = 0.05 * factor
        action[3:5] = 0.1 * factor
        
        # Smart capacitors
        if v_min < 0.94 or (v_min < 0.96 and trend < -0.001):
            action[10:12] = 1.0
            if v_min < 0.93:
                action[12] = 0.15
        elif v_min < 0.96:
            action[10] = 1.0
        
        # OLTC with trend
        if v_avg < 0.97 and trend <= 0:
            action[16] = 0.96
        elif v_avg > 1.03 and trend >= 0:
            action[16] = 1.04
        else:
            action[16] = 1.0
        
        return action


class SimpleL5_Optimal:
    """Near-optimal control."""
    def __init__(self):
        self.name = "L5_Optimal"
    
    def act(self, env):
        action = np.zeros(env.action_space.shape[0])
        
        sim = env.unwrapped.simulator
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        v_min = np.min(voltages)
        v_max = np.max(voltages)
        v_avg = np.mean(voltages)
        
        # Optimal renewable based on conditions
        if v_min < 0.95:
            # Emergency
            action[0:3] = 0.05   # Max renewable
            action[3:5] = 0.1
            action[10:12] = 1.0  # Big caps
            action[12] = 0.15    # Extra cap
            action[16] = 0.94    # Max OLTC boost
        elif v_avg < 0.98:
            # Low voltage
            action[0:3] = 0.04
            action[3:5] = 0.08
            action[10] = 1.0
            action[16] = 0.97
        elif v_avg > 1.02:
            # High voltage
            action[0:3] = 0.01
            action[3:5] = 0.02
            action[16] = 1.03
        else:
            # Normal - optimize for efficiency
            action[0:3] = 0.035  # 70% renewable
            action[3:5] = 0.07
            action[16] = 1.0
        
        return action


def collect_episode_data(env, controller, num_episodes=10):
    """Collect data from controller."""
    all_states = []
    all_actions = []
    all_rewards = []
    all_next_states = []
    all_dones = []
    
    episode_returns = []
    
    for episode in range(num_episodes):
        obs_tuple = env.reset()
        obs = obs_tuple[0] if isinstance(obs_tuple, tuple) else obs_tuple
        
        episode_return = 0
        
        # Vary load profile
        load_profiles = [1.0, 0.5, 1.2, 0.8, 1.1]
        env.load_scale = load_profiles[episode % len(load_profiles)]
        
        for t in range(100):
            action = controller.act(env)
            
            step_result = env.step(action)
            next_obs = step_result[0]
            reward = step_result[1]
            done = step_result[2]
            
            all_states.append(obs)
            all_actions.append(action)
            all_rewards.append(reward)
            all_next_states.append(next_obs)
            all_dones.append(done)
            
            episode_return += reward
            obs = next_obs
            
            if done:
                break
        
        episode_returns.append(episode_return)
    
    return {
        'states': np.array(all_states),
        'actions': np.array(all_actions),
        'rewards': np.array(all_rewards),
        'next_states': np.array(all_next_states),
        'dones': np.array(all_dones),
        'episode_returns': episode_returns
    }


def main():
    """Generate offline RL datasets."""
    print("="*80)
    print("GENERATING OFFLINE RL DATASETS")
    print("="*80)
    
    env = IEEE33ProperEnvironment()
    
    controllers = [
        SimpleL0_Random(),
        SimpleL1_Reactive(),
        SimpleL2_Proportional(),
        SimpleL3_Coordinated(),
        SimpleL4_Advanced(),
        SimpleL5_Optimal()
    ]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"offline_rl_data_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    summary = []
    
    for i, controller in enumerate(controllers):
        print(f"\nCollecting from {controller.name}...")
        
        num_episodes = 20 if i == 0 else 15  # More random exploration
        
        data = collect_episode_data(env, controller, num_episodes)
        
        avg_return = np.mean(data['episode_returns'])
        print(f"  Episodes: {num_episodes}")
        print(f"  Transitions: {len(data['rewards'])}")
        print(f"  Avg return: {avg_return:.4f}")
        
        # Add controller ID
        data['controller_id'] = i
        data['controller_name'] = controller.name
        
        # Save individual
        filename = f"{output_dir}/{controller.name}_data.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        all_data.append(data)
        summary.append({
            'controller': controller.name,
            'avg_return': avg_return,
            'transitions': len(data['rewards'])
        })
    
    # Create combined dataset
    print("\nCreating combined dataset...")
    combined = {
        'states': np.vstack([d['states'] for d in all_data]),
        'actions': np.vstack([d['actions'] for d in all_data]),
        'rewards': np.hstack([d['rewards'] for d in all_data]),
        'next_states': np.vstack([d['next_states'] for d in all_data]),
        'dones': np.hstack([d['dones'] for d in all_data]),
        'controller_ids': np.hstack([np.full(len(d['rewards']), d['controller_id']) for d in all_data])
    }
    
    with open(f"{output_dir}/combined_dataset.pkl", 'wb') as f:
        pickle.dump(combined, f)
    
    # Save summary
    with open(f"{output_dir}/summary.pkl", 'wb') as f:
        pickle.dump(summary, f)
    
    print(f"\nTotal transitions: {len(combined['rewards'])}")
    print(f"\nDatasets saved to: {output_dir}/")
    
    # Print summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    
    summary.sort(key=lambda x: x['avg_return'])
    for i, s in enumerate(summary):
        print(f"{i+1}. {s['controller']}: {s['avg_return']:.4f} avg return")
    
    print("\n✅ Offline RL dataset generation complete!")
    print(f"\nUse {output_dir}/combined_dataset.pkl for training")
    
    return output_dir


if __name__ == "__main__":
    output_dir = main()