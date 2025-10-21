"""Generate offline RL dataset with reactive power control."""

import numpy as np
import h5py
from create_algorithmic_diversity_with_q import (
    L0_Random, L1_BangBang, L2_Proportional,
    L3_PI_Controller, L4_RuleBasedExpert, L5_ScipyOptimal
)
from ready_to_use_l5_implementation import IEEE33ProperEnvironment


def generate_dataset(filename='dataset_with_q.h5', episodes_per_controller=20):
    """Generate dataset with all 6 controllers using reactive power."""

    controllers = [
        ('L0', L0_Random),
        ('L1', L1_BangBang),
        ('L2', L2_Proportional),
        ('L3', L3_PI_Controller),
        ('L4', L4_RuleBasedExpert),
        ('L5', L5_ScipyOptimal),
    ]

    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    all_terminals = []

    for ctrl_name, ctrl_class in controllers:
        print(f"\nGenerating data for {ctrl_name}...")

        for ep in range(episodes_per_controller):
            # Vary load scale across episodes - EXTREME loads to saturate discrete devices
            load_scale = np.random.uniform(4.0, 6.0)
            env = IEEE33ProperEnvironment(load_scale=load_scale)
            controller = ctrl_class(env)

            obs, _ = env.reset(seed=42 + ep)
            episode_reward = 0

            for step in range(96):
                action = controller.act(env)
                next_obs, reward, terminated, truncated, _ = env.step(action)

                all_obs.append(obs)
                all_actions.append(action)
                all_rewards.append(reward)
                all_next_obs.append(next_obs)

                # Mark terminal if this is last step OR environment terminated
                is_last_step = (step == 95)
                is_terminal = terminated or truncated or is_last_step
                all_terminals.append(1.0 if is_terminal else 0.0)

                episode_reward += reward
                obs = next_obs

                if terminated or truncated:
                    break

            print(f"  Episode {ep+1}/{episodes_per_controller}: reward={episode_reward:.2f}")

    # Convert to arrays
    obs = np.array(all_obs, dtype=np.float32)
    actions = np.array(all_actions, dtype=np.float32)
    rewards = np.array(all_rewards, dtype=np.float32)
    next_obs = np.array(all_next_obs, dtype=np.float32)
    terminals = np.array(all_terminals, dtype=np.float32)

    # Save to HDF5
    with h5py.File(filename, 'w') as f:
        g = f.create_group('transitions')
        g.create_dataset('obs', data=obs)
        g.create_dataset('action', data=actions)
        g.create_dataset('reward', data=rewards)
        g.create_dataset('next_obs', data=next_obs)
        g.create_dataset('terminal', data=terminals)

    print(f"\n✅ Dataset saved to {filename}")
    print(f"Total transitions: {len(obs)}")
    print(f"Mean reward: {rewards.mean():.3f}")
    print(f"Action dims 5-9 (Q) - mean: {actions[:, 5:10].mean():.4f}, std: {actions[:, 5:10].std():.4f}")

    return filename


if __name__ == "__main__":
    generate_dataset('dataset_with_q.h5', episodes_per_controller=20)
