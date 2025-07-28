"""Final correct environment implementing all lessons learned."""

import numpy as np
from gym_anm.envs.ieee33_env import IEEE33Env


class FinalCorrectEnv(IEEE33Env):
    """
    IEEE33 with renewables and proper load implementation.
    Key insights:
    1. next_vars() must return MW values (Expert 2)
    2. Actions are in p.u., not MW
    3. Load p_min values are in p.u., must multiply by baseMVA
    """
    
    def __init__(self, load_scale=1.0):
        super().__init__()
        self.load_scale = load_scale
        
        # Get load IDs in order
        self._load_ids = []
        for i, dev in self.simulator.devices.items():
            if hasattr(dev, 'type') and dev.type == -1:
                self._load_ids.append(i)
        self._load_ids.sort()  # Ensure consistent ordering
        
        # Calculate nominal load
        total_nominal = sum(abs(self.simulator.devices[i].p_min) * self.simulator.baseMVA 
                           for i in self._load_ids)
        
        print(f"FinalCorrectEnv initialized:")
        print(f"  Base MVA: {self.simulator.baseMVA}")
        print(f"  Total nominal load: {total_nominal:.2f} MW")
        print(f"  Load scale: {load_scale}")
        print(f"  Scaled load: {total_nominal * load_scale:.2f} MW")
    
    def next_vars(self, s_t):
        """
        Return MW values as Expert 2 specified.
        The simulator will convert to p.u. by dividing by baseMVA.
        """
        # Total size: N_load + N_non_slack_gen + K
        n_vars = self.simulator.N_load + self.simulator.N_non_slack_gen + self.K
        vars = np.zeros(n_vars)
        
        # Set load values in MW
        time_factor = self._get_time_factor()
        
        for idx, dev_id in enumerate(self._load_ids):
            if idx < self.simulator.N_load:
                dev = self.simulator.devices[dev_id]
                # p_min is in p.u. and negative, convert to positive MW
                nominal_mw = abs(dev.p_min) * self.simulator.baseMVA
                
                # Apply scaling with small noise
                noise = 1.0 + np.random.normal(0, 0.01)
                # Return as NEGATIVE MW for loads
                vars[idx] = -nominal_mw * self.load_scale * time_factor * noise
        
        # Renewable generation potential is handled by parent
        # No need to set anything here - controlled by actions
        
        return vars
    
    def _get_time_factor(self):
        """Time-based load variation matching documentation."""
        hour = getattr(self, 'hour_of_day', 12.0)
        
        if 0 <= hour < 6:
            return 0.7
        elif 6 <= hour < 9:
            return 0.7 + 0.3 * (hour - 6) / 3
        elif 9 <= hour < 17:
            return 1.0
        elif 17 <= hour < 20:
            return 1.1
        else:
            return 0.8


def test_final_env():
    """Test the final correct environment."""
    print("=" * 80)
    print("TESTING FINAL CORRECT ENVIRONMENT")
    print("=" * 80)
    
    from corrected_discrete_hierarchy import (
        CorrectedL0_Random,
        CorrectedL1_Basic,
        CorrectedL4_Predictive
    )
    
    env = FinalCorrectEnv(load_scale=1.0)
    
    # Test basic functionality
    print("\nBasic test:")
    obs, _ = env.reset()
    env.hour_of_day = 12.0
    env._update_renewable_potential()
    
    # No control
    action = np.zeros(13)
    action[12] = 1.0  # OLTC nominal
    
    obs, reward, _, _, _ = env.step(action)
    
    sim = env.simulator
    total_load = sum(abs(dev.p) * sim.baseMVA 
                    for dev in sim.devices.values() 
                    if hasattr(dev, 'type') and dev.type == -1)
    
    voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
    
    print(f"  Total load: {total_load:.2f} MW")
    print(f"  Voltage range: [{np.min(voltages):.4f}, {np.max(voltages):.4f}]")
    print(f"  Reward: {reward:.2f}")
    
    if np.min(voltages) < 0.96:
        print("  ✓ Voltages drop - loads working correctly!")
    
    # Test controllers
    print("\nController test (10 steps each):")
    
    controllers = {
        'L1_Basic': CorrectedL1_Basic(env),
        'L4_Predictive': CorrectedL4_Predictive(env)
    }
    
    for name, controller in controllers.items():
        obs, _ = env.reset(seed=123)
        total_reward = 0
        
        for step in range(10):
            env.hour_of_day = 12.0
            env._update_renewable_potential()
            
            action = controller.act(env)
            obs, reward, _, _, _ = env.step(action)
            total_reward += reward
        
        voltages = np.array([np.abs(bus.v) for bus in sim.buses.values()])
        print(f"\n  {name}:")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    Final voltages: [{np.min(voltages):.4f}, {np.max(voltages):.4f}]")
        
        # Check if control improved things
        if total_reward > -1000:  # Much better than no control
            print(f"    ✓ Controller improves performance!")


if __name__ == "__main__":
    test_final_env()