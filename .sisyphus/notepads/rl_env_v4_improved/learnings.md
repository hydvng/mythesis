Title: RL Env V4 Improved - Convergence Reward Addition

- Added _compute_multi_stage_convergence_reward(self, e, step) to implement multi-stage convergence rewards.
- Integrated convergence reward into step(): after computing base reward and smoothness reward, added convergence reward based on current error vector e and step count.
- Thresholds align with described design:
  - Final 20%: reward +5.0 when |z| < 5mm and |angles| < 1deg
  - Final 10%: reward +10.0 when tighter thresholds: |z| < 2.5mm and |angles| < 0.5deg
- Rationale: Encourages RL agent to converge more precisely in late-stage training, improving stability under relaxation of termination criteria.
- Validation plan: run longer horizon episodes and verify rewards reflect convergence events; ensure no regression in existing reward signals.
