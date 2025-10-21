"""Debug L5_ScipyOptimal to understand why it's losing to L1_BangBang."""

import numpy as np
from ready_to_use_l5_implementation import IEEE33ProperEnvironment
from create_algorithmic_diversity_with_q import L5_ScipyOptimal, L1_BangBang

def test_single_episode(controller, name, seed=42):
    """Test a controller for one episode."""
    env = IEEE33ProperEnvironment(load_scale=1.5)
    controller_instance = controller(env)

    obs, _ = env.reset(seed=seed)
    total_reward = 0
    violations = 0

    rewards = []

    for step in range(96):
        action = controller_instance.act(env)
        obs, reward, done, _, info = env.step(action)
        rewards.append(reward)
        total_reward += reward
        if info.get('dev_penalty', 0) > 0:
            violations += 1
        if done:
            break

    print(f"\n{name}:")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Avg reward: {total_reward/96:.6f}")
    print(f"  Violations: {violations}/96")
    print(f"  Min reward: {min(rewards):.6f}")
    print(f"  Max reward: {max(rewards):.6f}")

    return total_reward

def debug_objective_function():
    """Debug the L5 objective function to see what's happening."""
    env = IEEE33ProperEnvironment(load_scale=1.5)
    obs, _ = env.reset(seed=42)

    # Create L5 controller
    l5 = L5_ScipyOptimal(env)

    # Get first action
    print("\n" + "="*80)
    print("DEBUGGING L5 OBJECTIVE FUNCTION")
    print("="*80)

    # Take one step to see what happens
    action = l5.act(env)
    print(f"\nL5 first action:")
    print(f"  Renewable P: {action[0:5]}")
    print(f"  Renewable Q: {action[5:10]}")
    print(f"  Capacitors: {action[10:16]}")
    print(f"  OLTC: {action[16]}")

    # Step the environment with this action
    obs, reward, _, _, info = env.step(action)
    print(f"\nEnvironment response:")
    print(f"  Reward: {reward:.6f}")
    print(f"  Dev penalty: {info.get('dev_penalty', 0):.6f}")
    print(f"  E_loss: {info.get('e_loss', 0):.6f}")

    # Now test the objective function directly
    print("\n" + "="*80)
    print("TESTING OBJECTIVE FUNCTION DIRECTLY")
    print("="*80)

    # Reset for fresh state
    env.reset(seed=42)
    l5 = L5_ScipyOptimal(env)

    # Get simulator state
    import copy
    sim = env.unwrapped.simulator
    simulator_state = copy.deepcopy(sim)

    # Get p_pot values
    from gym_anm.simulator.components import RenewableGen, Load
    p_pot_values = []
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, RenewableGen):
            p_pot_values.append(dev.p_pot)

    # Build p_load_dict
    p_load_dict = {}
    for dev_id, dev in sim.devices.items():
        if isinstance(dev, Load):
            p_load_dict[dev_id] = dev.p

    # Test x0 (initial guess)
    x0 = np.zeros(17)
    voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
    v_min = voltages.min()

    if v_min < 0.95:
        x0[0:5] = l5.renewable_p_max * 0.8
        x0[5:10] = l5.renewable_q_max * 0.6
        x0[10:16] = [1, 1, 1, 1, 0, 0]
        x0[16] = 0.97
    elif v_min < 0.96:
        x0[0:5] = l5.renewable_p_max * 0.5
        x0[5:10] = l5.renewable_q_max * 0.4
        x0[10:16] = [1, 1, 0, 0, 0, 0]
        x0[16] = 0.99
    else:
        x0[0:5] = l5.renewable_p_max * 0.3
        x0[5:10] = l5.renewable_q_max * 0.2
        x0[10:16] = [0, 0, 0, 0, 0, 0]
        x0[16] = 1.0

    print(f"v_min = {v_min:.4f}")
    print(f"x0 (initial guess):")
    print(f"  Renewable P: {x0[0:5]}")
    print(f"  Renewable Q: {x0[5:10]}")
    print(f"  Caps (binary): {x0[10:16]}")
    print(f"  OLTC: {x0[16]}")

    # Evaluate objective at x0
    obj_value = l5.objective(x0, simulator_state, p_pot_values, p_load_dict)
    print(f"\nObjective at x0: {obj_value:.6f}")
    print(f"  (This is the COST to minimize, so reward = -{obj_value:.6f})")

    # Now step the environment with x0 action and compare
    env.reset(seed=42)
    action_from_x0 = np.zeros(17)
    action_from_x0[0:5] = x0[:5]
    action_from_x0[5:10] = x0[5:10]
    for i in range(6):
        if x0[10+i] > 0.5:
            action_from_x0[10+i] = l5.cap_ratings[i] / 10.0
    action_from_x0[16] = x0[16]

    obs, env_reward, _, _, info = env.step(action_from_x0)
    print(f"\nEnvironment reward for x0 action: {env_reward:.6f}")
    print(f"  Dev penalty: {info.get('dev_penalty', 0):.6f}")
    print(f"  E_loss: {info.get('e_loss', 0):.6f}")

    print(f"\n{'='*80}")
    print(f"COMPARISON:")
    print(f"  Objective value: {obj_value:.6f} (cost to minimize)")
    print(f"  Expected reward: {-obj_value:.6f}")
    print(f"  Actual env reward: {env_reward:.6f}")
    print(f"  MISMATCH: {abs(-obj_value - env_reward):.6f}")

    if abs(-obj_value - env_reward) > 0.001:
        print(f"\n>>> PROBLEM DETECTED: Objective function doesn't match environment!")
        print(f">>> This explains why scipy can't optimize properly.")
    else:
        print(f"\n>>> Objective function matches environment well.")

if __name__ == "__main__":
    # First debug the objective function
    debug_objective_function()

    # Then compare L1 vs L5
    print("\n" + "="*80)
    print("COMPARING L1 vs L5")
    print("="*80)

    l1_reward = test_single_episode(L1_BangBang, "L1_BangBang", seed=42)
    l5_reward = test_single_episode(L5_ScipyOptimal, "L5_ScipyOptimal", seed=42)

    print("\n" + "="*80)
    print("FINAL COMPARISON:")
    print(f"  L1_BangBang: {l1_reward:.4f}")
    print(f"  L5_ScipyOptimal: {l5_reward:.4f}")
    if l5_reward > l1_reward:
        print(f"  >>> L5 WINS by {l5_reward - l1_reward:.4f}")
    else:
        print(f"  >>> L1 WINS by {l1_reward - l5_reward:.4f} (NEEDS FIX)")
    print("="*80)
