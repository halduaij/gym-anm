# IEEE33 Offline RL Dataset Generation - Essential Files

This folder contains the essential files needed to generate offline RL datasets with algorithmic diversity for IEEE33 voltage control.

## Core Implementation Files

### 1. `create_algorithmic_diversity.py`
**Purpose:** Defines all 6 controller classes with different control algorithms
- **L0_Random**: Random control (worst, ~70% violations)
- **L1_BangBang**: Threshold-based control (if v < 0.96: MAX else MIN)
- **L2_Proportional**: P controller (action = Kp * error)
- **L3_PI_Controller**: PI controller with integral term
- **L4_RuleBasedExpert**: Rule-based system with trend analysis
- **L5_ScipyOptimal**: Mathematical optimization (best, 0 violations)

### 2. `ready_to_use_l5_implementation.py`
**Purpose:** Defines the IEEE33 power grid environment
- **IEEE33ProperEnvironment** class (inherits from gym-anm)
- 33-bus distribution network with 6 capacitor banks
- Action space: 17 dimensions (5 renewable + 6 capacitors + 6 OLTC)
- Proper reward structure: r = -(losses + violation_penalty)

### 3. `generate_final_offline_datasets.py`
**Purpose:** Main script to create offline RL datasets
- Runs each controller (L0-L5) for specified episodes
- Collects (state, action, reward, next_state) trajectories
- Saves as pickle files in `offline_rl_data_[timestamp]/`
- Creates combined dataset with all expertise levels

## Testing & Validation Files

### 4. `test_diverse_load_patterns.py`
**Purpose:** Validates controller diversity across realistic scenarios
- Tests 4 load patterns: baseline, renewable intermittence, industrial, extreme
- Generates comprehensive visualizations and performance reports
- Verifies algorithmic diversity is maintained across different conditions

### 5. `quick_dataset_test.py`
**Purpose:** Quick verification of generated datasets
- Loads and checks dataset integrity
- Prints basic statistics for validation

## Documentation

### 6. `CLAUDE.md`
**Purpose:** Complete project documentation
- Original requirements and goals
- Critical mistakes to avoid
- Expected performance hierarchy
- Implementation notes and updates

## Usage

To generate offline RL datasets:
```bash
source venv/bin/activate
python generate_final_offline_datasets.py
```

To test controller diversity:
```bash
python test_diverse_load_patterns.py
```

To verify L5 is optimal:
```bash
python quick_l3_vs_l5_test.py
```

## Performance Hierarchy (Expected)
1. L5 Scipy: ~-1.05 (best, 0 violations)
2. L3 PI: ~-1.56 (good, few violations)
3. L1 Bang-Bang: ~-3.40 (poor, many violations)
4. L2 Proportional: ~-4.83 (mediocre)
5. L4 Rule-Based: ~-70 (varies, edge cases)
6. L0 Random: ~-6000 (worst, ~70% violations)

Performance gap: ~5700x between best and worst!