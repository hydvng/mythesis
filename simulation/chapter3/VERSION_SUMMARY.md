# Chapter 3 RL Training Version Summary

## Overview

This document summarizes all RL training versions (V2-V5) for the ship-mounted stabilization platform control project.

---

## Version Comparison Table

| Version | Key Feature | Best Reward | Training Time | Status |
|---------|-------------|-------------|---------------|--------|
| **V2** | Fixed baseline issues | -0.34 | 265 min | ✅ Stable |
| **V3** | Curriculum learning | 1198.30 (ISE) | ~16 hours | ⚠️ Late divergence |
| **V4** | Warning penalty | 5622.62 | ~8 hours | ✅ Best overall |
| **V5 Simplified** | Pure ISE reward | -0.02 (ISE×10⁴) | 161 min | ✅ Most stable |

---

## V2: Fixed Baseline

### Key Fixes
1. **Action scale**: Changed from ±10000N to [-1, 1] with scale=5000
2. **Friction continuity**: Replaced `np.sign()` with `tanh()` approximation
3. **State normalization**: Adjusted for velocity overflow
4. **Initial state**: Fixed z∈[0.9, 1.1] (within constraints)
5. **Desired trajectory**: Changed z_des=1.0 (within constraints)

### Network Architecture
- Hidden layers: 256×4
- Parameters: ~256K
- Normalization: LayerNorm

### Training Results
- Episodes: 500
- Best reward: -0.34
- Constraint satisfaction: 100%

### Files
- Training: `experiments/train_improved_v2.py`
- Models: `experiments/models_improved/`
- Summary: `V2_IMPROVEMENTS_SUMMARY.md`

---

## V3: Curriculum Learning

### Approach
Three-stage curriculum:
1. **Stage 1**: 10s episodes, Hs=1.0m → 200 episodes
2. **Stage 2**: 20s episodes → 300 episodes  
3. **Stage 3**: 30s episodes → 380 episodes (diverged)

### Network Architecture
- Hidden layers: [512, 512, 256, 256]
- Parameters: ~470K
- Learning rate: critic 1e-3

### Results
| Stage | Episode Length | Best Reward | Notes |
|-------|----------------|-------------|-------|
| 1 | 10s | 399.79 | ✅ Fast convergence |
| 2 | 20s | 798.40 | ✅ Successful transfer |
| 3 | 30s | 1198.30 | ⚠️ Diverged @ Ep 340 |

### Key Insight
- Curriculum learning effectively transfers from shorter to longer episodes
- Stage 3 diverged due to high learning rate (1e-3 too aggressive for 30s)

### Files
- Training: `experiments/train_v3_curriculum.py`
- Models: `experiments/models_v3/`
- Summary: `V3_RESULTS_ANALYSIS.md`

---

## V4: Warning Penalty Mechanism

### Approach
Added warning_penalty mechanism to handle constraint violations:
- `diverge_threshold = 0.5`: Hard termination
- `warning_threshold = 0.4`: Soft penalty zone
- `warning_penalty = -5.0`: Penalty for warning zone

### Comparison Tested
| Variant | diverge_threshold | warning_threshold | Result |
|---------|------------------|-------------------|--------|
| V4 Optimized | 0.5 | N/A | Best overall |
| V4 Improved | 0.5 | 0.4 | Similar to baseline |
| Variant1 (Relaxed) | 1.0 | 0.4 | 38-74% worse |

### Key Findings
1. Warning mechanism provides no significant improvement
2. Relaxing thresholds actually **worsens** performance by 38-74%
3. Training-test mismatch: models trained at threshold=0.5 fail at 1.0

### Best Model
- **File**: `models_v4/stage2/Stage2_best.pt`
- **Reward**: 5,622.62
- **Episode**: 35 seconds

### Files
- Environment: `env/rl_env_v4_improved.py`
- Training: `experiments/v4_comparison/`
- Summary: `V4_IMPROVED_ANALYSIS_REPORT.md`

---

## V5 Simplified: Pure ISE Reward

### Motivation
Complex reward functions with multiple components create conflicting learning signals:
- Position penalty
- Velocity penalty  
- Control penalty
- ISE (Integral of Squared Error)
- Convergence bonus

### Solution
Simplified to single reward: `reward = -error²`

### Network Architecture
- Hidden layers: [512, 512, 256, 256]
- Parameters: ~1.8M
- Normalization: LayerNorm

### Training Results
| Metric | Value |
|--------|-------|
| Episodes | 400 |
| Best Reward | -0.02 |
| Best ISE | 0.000225 |
| Training Time | 161.2 minutes |

### Learning Curve
```
Episode 0:    Reward=-18.70, ISE=0.187
Episode 50:   Reward=-1.21,  ISE=0.012
Episode 100:  Reward=-0.34,  ISE=0.003
Episode 200:  Reward=-0.10,  ISE=0.001
Episode 300:  Reward=-0.07,  ISE=0.001
Episode 394:  Reward=-0.02,  ISE=0.000225 ← Best
```

### Key Insight
Pure ISE reward provides **stable, monotonic improvement** - no training instabilities observed.

### Control Performance (V5 Simplified)

#### Sinusoidal Trajectory
- ISE/step: 0.000011 (best among all versions)
- Excellent tracking performance

#### Constant Trajectory  
- ISE/step: 0.000001 (near-perfect)

### Files
- Environment: `env/rl_env_v5_simplified.py`
- Training: `experiments/v5_simplified/train_v5_simplified.py`
- Models: `experiments/v5_simplified/v5_simplified_training/`
- Plots: `experiments/v5_simplified/v5_control_plots/`

---

## Key Learnings

### What Works
1. ✅ LayerNorm over BatchNorm (BatchNorm fails with batch_size=1)
2. ✅ Moderate network size (1.8-2.6M optimal, 7.5M too large)
3. ✅ Simple reward functions (complex rewards cause instability)
4. ✅ Curriculum learning for progressive difficulty
5. ✅ Fixed thresholds (relaxing degrades performance)

### What Doesn't Work
1. ❌ Complex multi-component rewards (conflicting signals)
2. ❌ Warning penalty mechanisms (no benefit, adds complexity)
3. ❌ Relaxed termination thresholds (training-test mismatch)
4. ❌ Oversized networks (7.5M parameters too large for this task)
5. ❌ High learning rates for long episodes (1e-3 too aggressive)

---

## Recommendations

### For Chapter 3 Thesis
1. **Use V5 Simplified** for stable, interpretable results
2. **Report both** V4 (best overall reward) and V5 Simplified (most stable)
3. **Include learning curves** showing training stability differences

### Future Work
1. Try V5 Simplified with curriculum learning (V3 approach + V5 reward)
2. Experiment with different ISE weightings
3. Test on harder trajectories (higher frequency, amplitude)

---

*Generated: 2026-02-25*
*Project: Ship-mounted Stabilization Platform Control*
