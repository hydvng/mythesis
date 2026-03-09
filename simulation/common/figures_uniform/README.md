# uniform rod 仿真输出图说明

这些图由 `simulation/common/sim_uniform_rod_with_wave_demo.py` 自动生成。

## 图列表

- `uniform_position.png`
  - 位置对比：`q_s`（船体/扰动）、`q_c`（控制量）、`q=q_s+q_c`（上平台总位姿）
  - 单位：z 为 **mm**，角度为 **deg**。

- `uniform_velocity.png`
  - 速度对比：`qd_s`、`qd_c`、`qd=qd_s+qd_c`
  - 单位：z_dot 为 **mm/s**，角速度为 **deg/s**。

- `uniform_acceleration.png`
  - 加速度对比：`qdd_s`、`qdd_c`、`qdd=qdd_s+qdd_c`
  - 单位：z_ddot 为 **mm/s^2**，角加速度为 **deg/s^2**。

- `uniform_forces.png`
  - 广义力（每个 DOF 一张子图）：
    - `tau`：PD 控制器输出的广义力
    - `tau_s`：扰动等效广义力（按 $\tau_s = -M(q)\ddot q_s - C(q,\dot q)\dot q_s$）
    - `tau_total = tau + tau_s`
  - 单位：与模型定义一致（广义力/广义力矩混合量）。

- `uniform_disturbance_tau_s.png`
  - 仅绘制扰动等效广义力 `tau_s`（每个 DOF 一张子图）。

## 运行方式

从任意目录运行：

```bash
python3 /home/ubuntu/Documents/mythesis/simulation/common/sim_uniform_rod_with_wave_demo.py
```

默认仿真时长：100s（可在脚本 `main()` 中调整 `simulate(t_end=..., dt=...)`）。

## 当前控制器（减少抖动的 PID 版本）

`sim_uniform_rod_with_wave_demo.py` 当前默认使用“未知扰动”PID 控制器（不使用 $qdd_s$ 做扰动前馈）：

- PID（对总位姿误差 $e=q_{ref}-q$）：
  - `Kp = diag([2e4, 4e4, 4e4])`
  - `Ki = diag([3e3, 1.5e3, 1.5e3])`
  - `Kd = diag([2.2e4, 1.8e4, 1.8e4])`
- 误差一阶低通滤波：`err_filter_T=0.05s`
- 积分抗饱和：
  - `integral_limit=[0.5, 0.1, 0.1]`（分别对应 z/alpha/beta 的积分上限）
  - 若输出达到 `tau_limit`（限幅）则回退积分步进，避免 windup。

调参建议（降低抖动优先）：
- 抖动大：优先增大 `err_filter_T`（例如 0.08~0.15s），或适当减小 `Kd`。
- 稳态误差大：适当增大 `Ki`，并配合更严格的 `integral_limit`。

生成图片目录：`simulation/common/figures_uniform/`
