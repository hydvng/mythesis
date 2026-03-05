import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# 添加项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.dynamics_model import ThreeDOFDynamics
from controllers.observers.steso import STESO

def verify_steso():
    dt = 0.002
    total_time = 30.0
    steps = int(total_time / dt)
    t_arr = np.linspace(0, total_time, steps)
    
    # 1. 动力学模型 (Real Plant)
    dynamics = ThreeDOFDynamics()

    # ==================== 变负载/参数时变（研究工况） ====================
    # 说明：真实平台在海上应用很常见“载荷时变”（着舰/吊装/货物移动）。
    # 这里先实现最常用的两类：
    # - step:  t>=t0 时加载一个常值增量
    # - ramp:  在 [t0, t1] 内线性加载
    # 目前实现：只让质量 m_platform(t) 时变（最核心）；
    # 如希望更真实，可把 inertia_scale_with_mass=True，让 Ixx/Iyy 按比例同步变化。
    # 研究工况：开启时变负载（先不加偏心）
    enable_time_varying_payload = True
    payload_profile = "ramp"  # "step" | "ramp" | "none"
    payload_t0 = 8.0
    # 按需求：负载增加在 1s 内完成
    payload_t1 = 9.0
    payload_delta_mass = 100.0  # kg（按需求：考虑 100kg 变负载）
    inertia_scale_with_mass = True
    m0_nominal = float(dynamics.m_platform)
    Ixx0_nominal = float(dynamics.Ixx)
    Iyy0_nominal = float(dynamics.Iyy)

    def _payload_mass(t_query: float) -> float:
        if (not enable_time_varying_payload) or payload_profile == "none":
            return 0.0
        t_query = float(t_query)
        if payload_profile == "step":
            return payload_delta_mass if t_query >= payload_t0 else 0.0
        if payload_profile == "ramp":
            if t_query <= payload_t0:
                return 0.0
            if t_query >= payload_t1:
                return payload_delta_mass
            # linear
            return payload_delta_mass * (t_query - payload_t0) / max(1e-6, (payload_t1 - payload_t0))
        return 0.0

    # ==================== 重心偏置（研究工况） ====================
    # 说明：小幅重心偏置会带来“几乎常值”的重力力矩，从而导致姿态稳态偏差、腿力长期偏置，
    # 并进一步挤占速度/推力裕度。
    # 按需求：偏置半径 100mm。
    enable_cg_bias = True  
    cg_bias_radius = 0.1  # m (100mm)
    # 偏置方向：默认沿 +x（会主要激发 pitch 通道）；也可以改成 (x,y) 组合。
    cg_bias_dir = np.array([1.0, 1.0], dtype=float)
    # 按需求：偏心在 1s 内变化完成
    cg_bias_ramp_time = 1.0  # s

    def _cg_bias_xy(t_query: float, dm: float) -> np.ndarray:
        """返回重心偏置 (x_cg, y_cg) [m]。

        这里让重心偏置随着“载荷加载过程”同步出现：
        - 未加载时偏置为 0
        - 完全加载后偏置为给定半径
        这样更贴近“货物放上去导致重心偏移”的场景。
        """
        if (not enable_cg_bias) or cg_bias_radius <= 0.0:
            return np.zeros(2, dtype=float)
        # 按需求：偏心在 0.5s 内变化完成。
        # 这里以载荷开始时刻 payload_t0 作为偏心开始时刻。
        t_query = float(t_query)
        if (not enable_time_varying_payload) or payload_profile == "none":
            t_start = payload_t0
        else:
            t_start = payload_t0
        w = float(np.clip((t_query - t_start) / max(1e-6, cg_bias_ramp_time), 0.0, 1.0))
        d = cg_bias_dir / (np.linalg.norm(cg_bias_dir) + 1e-12)
        return w * cg_bias_radius * d
    
    # 2. 观测器 STESO
    # 设定扰动导数上界 delta (假设)
    # 假设扰动是正弦，W = A*sin(w*t)，则 W_dot = A*w*cos(w*t)，上界 delta = A*w
    # 设 A=5, w=2*pi*1 = 6.28 => delta ≈ 31.4
    # 增益条件: 
    #   beta1 > 2*delta => beta1 > 63
    #   beta2 需满足条件2
    # 为了测试方便，我们先选比较大的增益
    # 观察结果显示，Z轴扰动变化率最大 (~98)，Alpha (~49), Beta (~16)
    # 为避免"Lag"和"Clipping"，beta2 必须显著大于 max(W_dot)
    # 推荐 margin至少 3-4倍以保证良好的跟踪，特别是在离散系统中
    
    delta = 1.5  # 理论参考值
    
    # 针对性调参（Robust Tuning）
    # Condition: beta2 > delta * beta1 (approx)
    # delta=1.5, beta1=150 => beta2 > 225
    
    beta1 = 12.0
    beta2 = 30.0
    
    steso = STESO(dim=3, lambda1=4.0, beta1=beta1, beta2=beta2, dt=dt)
    
    # 3. 控制器
    # 当前脚本仅保留“任务空间 PID”（直接输出 tau_ctrl=[Fz,K,M]）。

    # 说明：已移除支链空间 PID，因此不再需要“腿推力限幅/推力变化率/行程/速度”约束统计。
    # --- 诊断：观察 z 轴为何跟不上 ---
    log_allocation_diagnostics = True
    jt_cond_max = 0.0
    jt_sigma_min_min = np.inf
    z_err_sum_sq = 0.0
    z_err_abs_max = 0.0

    # ==================== 执行器饱和（新增：力 & 速度/变化率） ====================
    # 目标：用最小改动在任务空间PID外面包一层“腿执行器约束”，观察饱和对 Z 精度的影响。
    # 实现策略：
    #   1) 先用当前构型 J^T 把任务空间 tau_cmd 映射为腿力 tau_legs_cmd（最小二乘）
    #   2) 对每条腿做：推力限幅 + 推力变化率限幅（用作“速度约束”的一阶近似）
    #   3) 用裁剪后的 tau_legs_act 通过 tau_act = J^T @ tau_legs_act 回到任务空间作为真实输入
    # 注意：这里的“速度饱和”没有直接建模电机转速/导程，只用 dF/dt 限制近似体现
    #       “高速工况下推力上不去/跟不上”的效果。
    enable_leg_actuator_limits = True
    leg_force_limit = 1000.0  # N, 单缸最大推/拉力（按你的 1000N）
    # 推力变化率上限：N/s。数值越小越“跟不住”，饱和影响越明显。
    # 这个需要结合你的电动缸/丝杆/电机功率来标定，这里先给一个能明显看出效果的保守值。
    # 若你关注的是“最大加速度”，推力变化率限幅会把系统永久钉在速饱和区，难以观察加速度约束效果。
    # 这里先放宽变化率限制（近似不限制），后续可再按电机功率标定。
    leg_force_rate_limit = 1.0e9  # N/s
    sat_force_hits = 0
    sat_rate_hits = 0
    sat_steps = 0

    # 腿力分配正则项：J^T 近奇异时，直接最小二乘可能给出很大的腿力，导致系统打满并失稳。
    # 用 Tikhonov 正则求解：min ||A x - b||^2 + reg*||x||^2
    allocation_reg = 1e-2

    # ==================== 最大加速度约束（用户指定） ====================
    # 这里将“最大加速度 15m/s^2”转成任务空间 Z 轴的最大等效力：|Fz| <= m * amax。
    # 注意：当前动力学里重力 G 会在系统方程中被减掉，因此 Fz 并不等于“总支撑力”。
    # 该限制更接近“控制输入允许产生的附加加速度幅值”。
    enable_task_accel_limit = True
    z_accel_limit = 15.0  # m/s^2

    # 用于每秒打印一次统计量
    print_every_steps = max(1, int(1.0 / dt))
    tau_legs_abs_max = np.zeros(3, dtype=float)
    tau_ctrl_abs_max = np.zeros(3, dtype=float)
    err_abs_max = np.zeros(3, dtype=float)
    
    # 4. 初始状态
    # 为避免初始时刻巨大的 PID 误差导致力饱和发散，将初始状态设为接近目标状态
    q = np.array([0.49, 0.1, 0.1]) # 初始位置 (与 q_des 一致)
    qd = np.zeros(3)

    # 4.1 观测器初始化（扰动估计初值）
    # 说明：你选择的“1”表示只改扰动估计初值 X_hat(0)，用于测试启动偏置/收敛鲁棒性。
    # 单位与 W_real/X_hat 一致（本脚本里相当于“加速度层面的总扰动”）。
    X_hat0 = np.array([0.0, 0.0, 0.0])
    steso.init_state(q, qd, q_des=np.array([0.5, 0.0, 0.0]), qd_des=np.zeros(3), X_hat0=X_hat0)
    
    # 存储历史数据
    history = {
        't': [],
        'w_dist_real': [],      # 加速度层面的等效扰动: H @ tau_ext
        'w_dist_hat': [],       # STESO 估计（加速度层面）: X_hat
        'w_total_real': [],     # 加速度层面的“总扰动”（含模型项）: H @ (tau_ext - C@qd - G)
        'w_total_hat': [],      # 加速度层面的“总扰动估计”: X_hat（与 w_total_real 语义对齐）
        'tau_dist_real': [],    # 力/力矩层面的真实扰动: tau_ext
        'tau_dist_hat': [],     # 力/力矩层面的估计扰动: M @ X_hat
        'tau_total_real': [],   # 力/力矩层面的“总扰动”等效广义力: tau_ext - C@qd - G
        'tau_total_hat': [],    # 力/力矩层面的“总扰动估计”等效广义力: M @ X_hat
        'S': [],
        'S_hat': [],
        'q': [],   # Added: system position
        'qd': [],  # Added: system velocity
        't_upper': [] # Added: convergence time upper bound
    }

    # 额外记录：三条支链推力时间序列（用于绘制“力曲线”）
    history['tau_legs'] = []

    # 额外记录：腿长/腿速（用于验证“先卡行程/速度”）
    history['l_legs'] = []
    history['ldot_legs'] = []
    l_prev = dynamics.compute_leg_lengths(q)

    # 记录：变负载曲线（便于对照误差/约束触发）
    history['payload_dm'] = []
    history['payload_m'] = []

    # 记录：重心偏置与其等效重力力矩
    history['cg_xy'] = []         # (N,2) in meters
    history['tau_cg'] = []        # (N,3) in [N, Nm, Nm] (only K/M used)

    # ==================== 任务空间 PID（还原） ====================
    # 说明：任务空间 PID 直接输出广义力/力矩 tau_ctrl = [Fz, K, M]
    # 先在“无偏心、无变负载”基准工况下把它调稳定，再逐步打开复杂工况。
    use_task_space_pid = True

    # 任务空间 PID 增益（初始给一组偏保守的值，避免一上来就饱和发散）
    kp_task = np.array([55000.0, 1500.0, 1500.0], dtype=float)   # [N/m, Nm/rad, Nm/rad]
    kd_task = np.array([16000.0, 300.0, 300.0], dtype=float)     # [N/(m/s), Nm/(rad/s), Nm/(rad/s)]
    ki_task = np.array([5000.0, 800.0, 800.0], dtype=float)        # [N/(m*s), Nm/(rad*s), Nm/(rad*s)]

    task_integral = np.zeros(3, dtype=float)
    task_integral_max = np.array([0.20, np.deg2rad(15.0), np.deg2rad(15.0)], dtype=float)

    # 任务空间输出限幅（先给比较宽松的上限，后续可按执行器能力收紧）
    enable_tau_task_saturation = True
    tau_task_limit = np.array([15000.0, 2500.0, 2500.0], dtype=float)

    # ==================== STESO 扰动补偿（新增：回注反馈） ====================
    # 语义：STESO 输出 X_hat 代表“加速度层面的等效总扰动”估计（单位与 qdd 同维度）。
    # 将其映射回力/力矩层面的等效扰动：tau_hat = M(q) @ X_hat
    # 控制律使用：tau_cmd = tau_pid - k_ff * tau_hat
    # 注意：这里补偿的是未建模项/外扰的等效广义力，对任务空间 PID 来说是最直接的“观测器反馈”。
    enable_steso_compensation = True
    steso_ff_gain = np.array([1.0, 1.0, 1.0], dtype=float)  # 可轴向调节
    
    # 设定用于收敛时间计算的保守 delta
    delta_est_for_bound = 1.5

    # 是否计算理论收敛时间上界（该函数内部会打印较多信息，长仿真会显著拖慢）
    enable_convergence_time_bound = False
    
    # 4. 仿真循环
    print("开始 STESO 验证仿真...")

    # ==================== 外扰输入：只使用一个扰动文件（平台外载） ====================
    # 约定：NPZ 包含
    #   - t: (N,)
    #   - tau_ext: (N,3) with columns [Fz(N), K(Nm), M(Nm)]
    tau_ext_provider = None  # callable(t)->(3,)
    platform_tau_npz = os.path.join(os.path.dirname(__file__), "../disturbances", "platform_sea_state4_tau_ext.npz")
    if os.path.exists(platform_tau_npz):
        d = np.load(platform_tau_npz, allow_pickle=True)
        tau_t = np.asarray(d["t"], dtype=float).reshape(-1)
        tau_ext_arr = np.asarray(d["tau_ext"], dtype=float)
        if tau_ext_arr.ndim != 2 or tau_ext_arr.shape[1] != 3:
            raise ValueError(f"{platform_tau_npz} 中 tau_ext 的形状应为 (N,3)，实际为 {tau_ext_arr.shape}")

        def _tau_ext_from_platform(t_query: float) -> np.ndarray:
            t_query = float(t_query)
            out = np.zeros(3, dtype=float)
            for k in range(3):
                out[k] = np.interp(t_query, tau_t, tau_ext_arr[:, k], left=tau_ext_arr[0, k], right=tau_ext_arr[-1, k])
            return out

        tau_ext_provider = _tau_ext_from_platform
        print(f"已加载扰动文件: {os.path.abspath(platform_tau_npz)}")
    else:
        print("未找到 platform_sea_state4_tau_ext.npz，回退到正弦外扰 tau_ext。")

    # ==================== 观测噪声（默认关闭，不改变原脚本行为） ====================
    # 说明：噪声只注入到“观测量/测量量”，不影响真实系统演化。
    # - 关闭时（enable_measurement_noise=False），脚本输出应与原脚本一致。
    # - 开启时，STESO 与控制器将看到带噪声的 q/qd。
    # ✅ 按需求：开启测量噪声，并固定噪声量级（Z 向 5e-3 m）
    # 注意：如果后续又看到噪声“没生效”，优先检查这里是否被格式化/其他编辑改回。
    enable_measurement_noise = False
    # 标准差（可按需要调整；这里给一个“明显但仍合理”的默认量级）
    # - Z:     5 mm 量级（中等精度位移传感/视觉在轻微抖动下的可见噪声）
    # - 角度:  0.5 deg 量级（IMU/姿态解算在海况/振动下的可见噪声）
    # - 速度:  对应更“抖”的导数估计，这里给到 0.05 m/s、2 deg/s 量级
    sigma_q  = np.array([5e-3, np.deg2rad(0.5), np.deg2rad(0.5)])   # [m, rad, rad]
    sigma_qd = np.array([5e-2, np.deg2rad(2.0), np.deg2rad(2.0)])   # [m/s, rad/s, rad/s]
   
    # sigma_q  = np.array([0.0, 0.0, 0.0])   # [m, rad, rad]
    # sigma_qd = np.array([0.0, 0.0, 0.0])   # [m/s, rad/s, rad/s]
    rng = np.random.default_rng(42)

    def _measure_state(q_true: np.ndarray, qd_true: np.ndarray):
        if not enable_measurement_noise:
            return q_true, qd_true
        q_m = q_true + rng.normal(0.0, sigma_q, size=3)
        qd_m = qd_true + rng.normal(0.0, sigma_qd, size=3)
        return q_m, qd_m

    # ==================== 一阶低通滤波（对测量值） ====================
    # 目的：在“有噪声”时降低 q/qd 的高频抖动，观察 STESO 的估计抖振是否改善。
    # 离散形式：y[k] = y[k-1] + alpha * (x[k] - y[k-1])
    # 其中 alpha = dt / (tau + dt)，tau 越大滤波越强但延迟越大。
    enable_measurement_filter = True
    # 在 5mm 测量噪声下，不滤波会导致 qd 噪声很大，从而 kd 放大噪声引起 z 波动。
    # 这里稍微加重滤波（仍保持毫秒级），通常能明显降低 z 抖动。
    tau_filter = 0.02  # s
    alpha_filter = dt / (tau_filter + dt)

    q_filt = q.copy()
    qd_filt = qd.copy()

    def _filter_measurement(q_in: np.ndarray, qd_in: np.ndarray):
        nonlocal q_filt, qd_filt
        if not enable_measurement_filter:
            return q_in, qd_in
        q_filt = q_filt + alpha_filter * (q_in - q_filt)
        qd_filt = qd_filt + alpha_filter * (qd_in - qd_filt)
        return q_filt, qd_filt

    # ==================== 时滞建模（可选） ====================
    # 常见时滞通常出现在两个环节：
    # 1) 传感/估计链路：测量→通讯→滤波/融合→控制器（measurement delay）
    # 2) 执行链路：控制器→驱动器→执行器建立力/力矩（actuator/command delay）
    #
    # 这里用最简单、可复现的离散“纯时延”模型：FIFO 队列（ZOH + N 步延迟）。
    # - enable_* = False 时完全不改变原脚本行为
    # - delay_seconds 会被量化为 delay_steps = round(delay_seconds/dt)
    enable_measurement_delay = False
    measurement_delay_seconds = 0.002  # e.g. 0.01 -> 10ms

    enable_actuator_delay = False
    actuator_delay_seconds = 0.01  # e.g. 0.01 -> 10ms

    def _delay_steps(delay_s: float) -> int:
        return int(max(0, round(float(delay_s) / max(1e-12, dt))))

    meas_delay_steps = _delay_steps(measurement_delay_seconds) if enable_measurement_delay else 0
    act_delay_steps = _delay_steps(actuator_delay_seconds) if enable_actuator_delay else 0

    # FIFO 初始化：用初值填充，避免仿真起始阶段出现“空队列”。
    _q_meas_fifo = [q.copy() for _ in range(max(1, meas_delay_steps + 1))]
    _qd_meas_fifo = [qd.copy() for _ in range(max(1, meas_delay_steps + 1))]
    _tau_ctrl_fifo = [np.zeros(3, dtype=float) for _ in range(max(1, act_delay_steps + 1))]
    
    for i in range(steps):
        t = t_arr[i]

        # --- 时变负载：更新真实系统参数（质量/惯量） ---
        dm = _payload_mass(t)
        dynamics.m_platform = m0_nominal + dm
        if inertia_scale_with_mass:
            # 简单假设：惯量与质量按比例变化（保持几何尺寸不变时是合理的一阶近似）
            scale_I = (m0_nominal + dm) / max(1e-9, m0_nominal)
            dynamics.Ixx = Ixx0_nominal * scale_I
            dynamics.Iyy = Iyy0_nominal * scale_I

        history['payload_dm'].append(dm)
        history['payload_m'].append(float(dynamics.m_platform))

        # Record states before update (or after, just being consistent)
        history['q'].append(q.copy())
        history['qd'].append(qd.copy())
        
        # --- A. 生成"真实"外部扰动 tau_ext ---
        # 模拟一个随时间变化的外部力矩/力 (例如正弦波)
        # 注意：W 是加速度层面的扰动，W = M^-1(tau_ext - C*qd - G) 如果按标准推导
        # 但在STESO的公式里，W 被定义为 "总扰动"，包括了模型不确定性。
        # 为了精确验证，我们手动构造 dynamics 的输出，使其符合 dot_S = H*tau + F + W_desired
        
        # 简单起见，我们直接模拟物理系统，然后反算 W_real
        
        # 构造外部力 tau_ext
        if tau_ext_provider is not None:
            tau_ext = tau_ext_provider(t)
        else:
            # 回退：手工正弦外扰（用于无数据时的快速自检）
            freq = 0.5 # Hz
            amp = 1.0  # N/Nm
            tau_ext = np.array([
                amp * 0.5 * np.sin(2*np.pi*freq*t),
                amp * 0.1 * np.cos(2*np.pi*freq*t),
                amp * 0.1 * np.sin(2*np.pi*0.5*t)
            ])

        # --- 重心偏置引入的“等效重力力矩”叠加到外扰 ---
        # 小角度下，重力力矩可近似：
        #   tau_alpha ≈ +m*g*y_cg
        #   tau_beta  ≈ -m*g*x_cg
        cg_xy = _cg_bias_xy(t, dm)  # [x_cg, y_cg]
        tau_cg = np.zeros(3, dtype=float)
        tau_cg[1] = float(dynamics.m_platform * dynamics.g * cg_xy[1])
        tau_cg[2] = float(-dynamics.m_platform * dynamics.g * cg_xy[0])
        tau_ext = tau_ext + tau_cg

        history['cg_xy'].append(cg_xy.copy())
        history['tau_cg'].append(tau_cg.copy())
        
        # 控制输入 tau
        # 使用 PIDController 计算支链力，然后映射回任务空间
        q_des = np.array([0.5, 0.0, 0.0])
        qd_des = np.zeros(3)

        # --- A0. 生成测量值（用于控制器/观测器输入） ---
        # 顺序建议：先生成带噪测量 → 再滤波 → 再加“纯时延”（更符合工程：滤波/融合也在测量链路里）
        q_meas_raw, qd_meas_raw = _measure_state(q, qd)
        q_meas_f, qd_meas_f = _filter_measurement(q_meas_raw, qd_meas_raw)

        if enable_measurement_delay and meas_delay_steps > 0:
            _q_meas_fifo.append(q_meas_f.copy())
            _qd_meas_fifo.append(qd_meas_f.copy())
            q_meas = _q_meas_fifo.pop(0)
            qd_meas = _qd_meas_fifo.pop(0)
        else:
            q_meas, qd_meas = q_meas_f, qd_meas_f

        # --- A1. 控制律：任务空间 PID（唯一控制器） ---
        # 输出 tau_ctrl = [Fz, K, M]
        e_task_ctrl = q_des - q_meas
        ed_task_ctrl = qd_des - qd_meas

        task_integral += e_task_ctrl * dt
        task_integral = np.clip(task_integral, -task_integral_max, task_integral_max)

        tau_pid = kp_task * e_task_ctrl + kd_task * ed_task_ctrl + ki_task * task_integral

        # --- A1.1 STESO 扰动补偿回注 ---
        # 用“当前构型的 M”把加速度层面的估计映射为等效广义力/力矩，并做前馈抵消。
        # 这里用 q_meas/qd_meas 计算更贴近控制回路的测量输入。
        tau_hat = np.zeros(3, dtype=float)
        if enable_steso_compensation:
            try:
                M_meas = dynamics.mass_matrix(q_meas)
                tau_hat = M_meas @ steso.X_hat
            except Exception:
                tau_hat = np.zeros(3, dtype=float)

        tau_cmd = tau_pid - steso_ff_gain * tau_hat

        # --- A1.1.5 最大加速度约束（Z 轴）---
        if enable_task_accel_limit:
            fz_lim = float(dynamics.m_platform * z_accel_limit)
            tau_cmd[0] = float(np.clip(tau_cmd[0], -fz_lim, fz_lim))

        if enable_tau_task_saturation:
            tau_cmd = np.clip(tau_cmd, -tau_task_limit, tau_task_limit)

        # --- A1.2 执行器（腿）约束：把 tau_cmd 变成实际可实现的 tau_ctrl ---
        # 用测量构型做分配，符合“控制器看到的构型”。
        J_curr_for_log = dynamics.jacobian(q_meas)
        A = J_curr_for_log.T
        b = tau_cmd
        try:
            AtA = A.T @ A
            tau_legs_cmd = np.linalg.solve(AtA + allocation_reg * np.eye(AtA.shape[0]), A.T @ b)
        except Exception:
            try:
                tau_legs_cmd = np.linalg.lstsq(A, b, rcond=None)[0]
            except Exception:
                tau_legs_cmd = np.zeros(3, dtype=float)

        tau_legs_act = tau_legs_cmd.copy()
        saturated_this_step = False
        if enable_leg_actuator_limits:
            sat_steps += 1

            # 1) 推力限幅
            before = tau_legs_act.copy()
            tau_legs_act = np.clip(tau_legs_act, -leg_force_limit, leg_force_limit)
            if np.any(np.abs(before - tau_legs_act) > 1e-9):
                sat_force_hits += 1
                saturated_this_step = True

            # 2) 推力变化率限幅（等效“速度/功率跟不上”）
            # 使用上一时刻实际腿力作为参考。
            if i == 0:
                tau_legs_prev_act = tau_legs_act.copy()
            else:
                tau_legs_prev_act = history['tau_legs'][-1]
            d_tau = tau_legs_act - tau_legs_prev_act
            d_tau_lim = leg_force_rate_limit * dt
            d_tau_clipped = np.clip(d_tau, -d_tau_lim, d_tau_lim)
            tau_legs_act = tau_legs_prev_act + d_tau_clipped
            if np.any(np.abs(d_tau - d_tau_clipped) > 1e-9):
                sat_rate_hits += 1
                saturated_this_step = True

        # 积分抗饱和：若触发腿执行器饱和，撤销本步积分（等效冻结积分）
        if enable_leg_actuator_limits and saturated_this_step:
            task_integral -= e_task_ctrl * dt

        # 真实作用到平台的控制输入（任务空间）
        tau_ctrl = J_curr_for_log.T @ tau_legs_act

        # --- A1.4 将最大加速度约束施加到“真实输入” ---
        # 目的：即使 PID/分配给出过大的 Fz，也保证系统实际 Z 加速度不超过上限。
        # 做法：用当前真实状态(q,qd)估计 qdd，并对 qdd_z 裁剪，然后回推需要的 Fz。
        if enable_task_accel_limit:
            try:
                M_true = dynamics.mass_matrix(q)
                C_true = dynamics.coriolis_matrix(q, qd)
                G_true = dynamics.gravity_vector(q)
                H_true = np.linalg.inv(M_true)
                qdd_est = H_true @ (tau_ctrl + tau_ext - C_true @ qd - G_true)
                qdd_z_limited = float(np.clip(qdd_est[0], -z_accel_limit, z_accel_limit))
                # 只修改 Z 通道等效力，使得估计的 qdd_z 满足上限
                tau_ctrl = tau_ctrl.copy()
                tau_ctrl[0] = float((M_true[0, :] @ np.array([qdd_z_limited, qdd_est[1], qdd_est[2]])) - tau_ext[0] + (C_true @ qd)[0] + G_true[0])
            except Exception:
                pass

        # --- A1.5 执行链路时滞（可选）：让“作用到被控对象的输入”滞后于控制器输出 ---
        # 注意：这会直接降低相位裕度，通常比测量延迟更“致命”。建议从 2~10ms 小量开始。
        if enable_actuator_delay and act_delay_steps > 0:
            _tau_ctrl_fifo.append(tau_ctrl.copy())
            tau_ctrl = _tau_ctrl_fifo.pop(0)

        # 用于记录/画图：我们记录“实际腿力”（限幅后的腿力）
        tau_legs = tau_legs_act

        # --- 诊断统计 ---
        if log_allocation_diagnostics:
            try:
                svals = np.linalg.svd(J_curr_for_log.T, compute_uv=False)
                sigma_min = float(np.min(svals))
                sigma_max = float(np.max(svals))
                cond = float(sigma_max / (sigma_min + 1e-12))
                jt_cond_max = max(jt_cond_max, cond)
                jt_sigma_min_min = min(jt_sigma_min_min, sigma_min)
            except Exception:
                pass

        # z 轴误差统计
        z_err = float(q_meas[0] - q_des[0])
        z_err_sum_sq += z_err * z_err
        z_err_abs_max = max(z_err_abs_max, abs(z_err))

        # 统计量（验证 PID 是否真的在影响 tau）
        e_task = q_meas - q_des
        tau_legs_abs_max = np.maximum(tau_legs_abs_max, np.abs(tau_legs))
        tau_ctrl_abs_max = np.maximum(tau_ctrl_abs_max, np.abs(tau_ctrl))
        err_abs_max = np.maximum(err_abs_max, np.abs(e_task))

        if (i % print_every_steps) == 0 and i > 0:
            print(
                f"t={t:6.1f}s | |e|_max=[{err_abs_max[0]:.3f}m, {np.rad2deg(err_abs_max[1]):.2f}deg, {np.rad2deg(err_abs_max[2]):.2f}deg] "
                f"| |tau_legs|_max=[{tau_legs_abs_max[0]:.0f},{tau_legs_abs_max[1]:.0f},{tau_legs_abs_max[2]:.0f}]N "
                f"| |tau_ctrl|_max=[{tau_ctrl_abs_max[0]:.0f},{tau_ctrl_abs_max[1]:.0f},{tau_ctrl_abs_max[2]:.0f}]"
            )
        
        # --- B. 物理步进 (Real World) ---
        # 计算 M, C, G
        M = dynamics.mass_matrix(q)
        C = dynamics.coriolis_matrix(q, qd)
        G = dynamics.gravity_vector(q)
        H = np.linalg.inv(M)
        
        # 真实加速度 qdd = M^-1 * (tau_ctrl + tau_ext - C*qd - G)
        # qdd = H @ (tau_ctrl + tau_ext - C @ qd - G)
        
        # RK4 积分
        def system_dynamics(q_val, qd_val):
            # Compute dynamics terms at current state
            M_val = dynamics.mass_matrix(q_val)
            C_val = dynamics.coriolis_matrix(q_val, qd_val)
            G_val = dynamics.gravity_vector(q_val)
            H_val = np.linalg.inv(M_val)
            
            # Recompute external force (approx constant in sub-step or time-dependent)
            # ideally tau_ext should depend on t, but for small dt we use current t value
            # same for tau_ctrl (Hold Zero Order)
            
            acc = H_val @ (tau_ctrl + tau_ext - C_val @ qd_val - G_val)
            return qd_val, acc

        # k1
        k1_q, k1_qd = system_dynamics(q, qd)
        
        # k2
        k2_q, k2_qd = system_dynamics(q + 0.5*dt*k1_q, qd + 0.5*dt*k1_qd)
        
        # k3
        k3_q, k3_qd = system_dynamics(q + 0.5*dt*k2_q, qd + 0.5*dt*k2_qd)
        
        # k4
        k4_q, k4_qd = system_dynamics(q + dt*k3_q, qd + dt*k3_qd)
        
        # Update
        q_next = q + (dt/6.0)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
        qd_next = qd + (dt/6.0)*(k1_qd + 2*k2_qd + 2*k3_qd + k4_qd)
        
        # --- C. 扰动真值定义（推荐分层看） ---
        # 1) 力/力矩层面扰动（物理直观）：tau_dist_real = tau_ext
        # 2) 加速度层面“等效扰动”（供 STESO 对比）：w_dist_real = H @ tau_ext
        # 3) “总扰动”（更贴近 STESO 在很多推导里的语义）：
        #    w_total_real = H @ (tau_ext - C@qd - G)
        #    它把模型项（科氏/重力）也并入了“等效扰动”，因此常见到 Z 方向近似常值（接近 -g）。
        #
        # ✅ 重要：当前 STESO 实现的已知项里没有显式加入 -H(C@qd + G)，
        # 要想和 X_hat 的“总扰动估计”对齐，比较 w_total_real 更直观。
        #
        # 说明：之前版本使用 H @ (tau_ext - C@qd - G)，会把模型本身的 C/G 也算进“扰动”，
        # 容易造成单位/物理含义混乱（尤其你现在关心的是“力和力矩”层面）。
        tau_dist_real = tau_ext
        w_dist_real = H @ tau_ext

        tau_total_real = tau_ext - (C @ qd) - G
        w_total_real = H @ tau_total_real

        # --- D. 运行观测器 ---
        # 观测器输入: q, qd, tau_ctrl, H
        # 注意: 这里的 H 应该是观测器认为的模型参数。
        # 为了单纯验证观测器算法，先假设观测器拥有完美的 H (M^-1)。
        # 如果观测器只知道近似的 M_hat，那么误差会更大，这里先测理想情况。
        X_hat = steso.update(q_meas, qd_meas, tau_ctrl, H, q_des=q_des)
        tau_dist_hat = M @ X_hat
        # 为了保持命名一致：STESO 估计的 X_hat 在这里也当作“总扰动估计”（加速度层面）
        w_total_hat = X_hat
        tau_total_hat = tau_dist_hat

        # 计算当前的理论收敛时间上界
        # S = lambda1 * e + ed
        # e = q - q_des, ed = qd - qd_des
        e_curr = q_meas - q_des
        ed_curr = qd_meas - qd_des # (Assuming qd_des is 0)
        S_curr = steso.lambda1 * e_curr + ed_curr

        if enable_convergence_time_bound:
            t_bound, is_stable, gamma_val = steso.calculate_convergence_upper_bound(
                S_curr, steso.S_hat, w_dist_real, X_hat, delta_est_for_bound
            )
            history['t_upper'].append(t_bound if is_stable else np.nan)
        else:
            history['t_upper'].append(np.nan)

        # 记录
        history['t'].append(t)
        history['w_dist_real'].append(w_dist_real.copy())
        history['w_dist_hat'].append(X_hat.copy())
        history['tau_dist_real'].append(tau_dist_real.copy())
        history['tau_dist_hat'].append(tau_dist_hat.copy())
        history['w_total_real'].append(w_total_real.copy())
        history['w_total_hat'].append(w_total_hat.copy())
        history['tau_total_real'].append(tau_total_real.copy())
        history['tau_total_hat'].append(tau_total_hat.copy())
        history['tau_legs'].append(tau_legs.copy())

        # 腿长/腿速记录（用于验证“先卡行程/速度”）
        l_curr_true = dynamics.compute_leg_lengths(q)
        ldot_curr = (l_curr_true - l_prev) / dt
        l_prev = l_curr_true
        history['l_legs'].append(l_curr_true.copy())
        history['ldot_legs'].append(ldot_curr.copy())

        # 更新状态
        q = q_next
        qd = qd_next

    # 5. 验证参数选择条件
    
    # 计算实际的 W_real 统计 (Per Axis)
    W_real_arr = np.array(history['w_dist_real'])
    W_dot_approx = np.gradient(W_real_arr, dt, axis=0)
    
    # Per-axis delta
    delta_per_axis = np.max(np.abs(W_dot_approx), axis=0)
    delta_actual = np.max(delta_per_axis)
    
    print("\n" + "="*50)
    print("信号分析 (Signal Analysis Per Axis)")
    axes_names = ['Z', 'Alpha', 'Beta']
    for k, name in enumerate(axes_names):
        print(f"Axis {name}: Max W_dot = {delta_per_axis[k]:.4f}")
    
    print("-" * 30)
    print(f"Global Max W_dot (delta_actual): {delta_actual:.4f}")
    print("="*50)

    print("参数选择验证 (Parameter Verification)")
    print("="*50)
    print(f"设定参数: beta1 = {beta1}, beta2 = {beta2}")
    
    # 验证条件 1: beta1 > 2 * delta
    cond1_limit = 2 * delta_actual
    is_cond1_ok = beta1 > cond1_limit
    print(f"\n[条件 1] beta1 > 2*delta ({cond1_limit:.4f})")
    print(f"  -> Result: {'✅ PASS' if is_cond1_ok else '❌ FAIL'}")
    if not is_cond1_ok:
        print(f"  建议: 增加 beta1 至少到 {cond1_limit * 1.1:.1f}")

    # 验证条件 2: Q矩阵正定性 (Determinant > 0)
    # Q1 = [[ b1*b2 + 0.5*b1^3 - delta*(b2+1),  -0.5*b1^2 ],
    #       [ -0.5*b1^2,                        0.5*b1 - delta ]]
    # Determinant approx: 0.5*beta1^2 * (beta2 - delta*beta1) > 0
    # Strict check:
    delta_actual = delta_est_for_bound
    q11 = beta1 * beta2 + 0.5 * beta1**3 - delta_actual * (beta2 + 1)
    q12 = -0.5 * beta1**2
    q22 = 0.5 * beta1 - delta_actual
    
    det_Q = q11 * q22 - q12**2
    
    print(f"\n[条件 2] Q矩阵正定性 (Det > 0)")
    print(f"  Det(Q) = {det_Q:.4e}")
    if det_Q > 0 and q11 > 0 and q22 > 0:
         print("  -> Result: ✅ PASS")
    else:
         print("  -> Result: ❌ FAIL")
         rec_beta2 = delta_actual * beta1
         print(f"  建议: beta2 需满足 beta2 > delta * beta1 (约 {rec_beta2:.1f})")

    print("="*50 + "\n")

    # 控制/分配诊断汇总
    z_rmse = np.sqrt(z_err_sum_sq / max(1, steps))
    print("控制/分配诊断 (Control & Allocation Diagnostics)")
    print("="*50)
    print(f"J^T cond max      : {jt_cond_max:.3e}")
    print(f"J^T sigma_min min : {jt_sigma_min_min:.3e}")
    print(f"Z error RMSE      : {z_rmse:.4f} m")
    print(f"Z error abs max   : {z_err_abs_max:.4f} m")
    if enable_leg_actuator_limits and sat_steps > 0:
        print("\n执行器饱和统计 (Actuator Saturation Stats)")
        print("-"*30)
        print(f"Force saturation steps : {sat_force_hits}/{sat_steps} ({100.0*sat_force_hits/sat_steps:.1f}%)")
        print(f"Rate  saturation steps : {sat_rate_hits}/{sat_steps} ({100.0*sat_rate_hits/sat_steps:.1f}%)")
    print("="*50 + "\n")

    # 说明：已移除支链空间 PID，因此不输出约束触发统计（sat_counts）。

    # 收敛时间分析
    print("理论收敛时间分析 (Convergence Time Analysis)")
    t_upper_arr = np.array(history['t_upper'])
    valid_bounds = t_upper_arr[~np.isnan(t_upper_arr)]
    
    if len(valid_bounds) > 0:
        print(f"Initial Convergence Time Bound: {valid_bounds[0]:.4f} s")
        print(f"Max Convergence Time Bound: {np.max(valid_bounds):.4f} s")
        print(f"Mean Convergence Time Bound: {np.mean(valid_bounds):.4f} s")
        print(f"Final Convergence Time Bound: {valid_bounds[-1]:.4f} s")
    else:
        print("无法计算收敛时间 (参数不稳定或 delta 设置不合理)")
    print("="*50 + "\n")

    # 5. 绘图
    history['t'] = np.array(history['t'])
    history['w_dist_real'] = np.array(history['w_dist_real'])
    history['w_dist_hat'] = np.array(history['w_dist_hat'])
    history['w_total_real'] = np.array(history['w_total_real'])
    history['w_total_hat'] = np.array(history['w_total_hat'])
    history['tau_dist_real'] = np.array(history['tau_dist_real'])
    history['tau_dist_hat'] = np.array(history['tau_dist_hat'])
    history['tau_total_real'] = np.array(history['tau_total_real'])
    history['tau_total_hat'] = np.array(history['tau_total_hat'])
    history['q'] = np.array(history['q'])
    history['qd'] = np.array(history['qd'])
    history['tau_legs'] = np.array(history['tau_legs'])
    history['l_legs'] = np.array(history['l_legs'])
    history['ldot_legs'] = np.array(history['ldot_legs'])
    history['payload_dm'] = np.array(history['payload_dm'])
    history['payload_m'] = np.array(history['payload_m'])
    history['cg_xy'] = np.array(history['cg_xy'])
    history['tau_cg'] = np.array(history['tau_cg'])
    
    print("仿真完成，正在绘图...")
    
    # Plot 1: 扰动力/力矩（物理层面） + 收敛时间上界
    plt.figure(figsize=(10, 14))
    titles = ['Z', 'Alpha', 'Beta']
    tau_units = ['N', 'N·m', 'N·m']

    for i in range(3):
        plt.subplot(4, 1, i+1)
        tau_real_plot = history['tau_dist_real'][:, i]
        tau_hat_plot = history['tau_dist_hat'][:, i]

        plt.plot(history['t'], tau_real_plot, 'k-', label='Real Disturbance (tau_ext)', linewidth=2, alpha=0.6)
        plt.plot(history['t'], tau_hat_plot, 'r--', label='Estimated Disturbance (M @ X_hat)', linewidth=1.5)
        plt.title(f'Disturbance (Force/Torque) - {titles[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Amplitude ({tau_units[i]})')
        plt.grid(True)
        plt.legend()

    # Plot 4: Convergence Time Bound
    plt.subplot(4, 1, 4)
    plt.plot(history['t'], history['t_upper'], 'g-', label='Theoretical Time Bound (t_s1)', linewidth=2)
    plt.title('Theoretical Convergence Time Upper Bound Distribution')
    plt.xlabel('Time (s)')
    plt.ylabel('Time Bound (s)')
    plt.grid(True)
    plt.legend()
        
    plt.tight_layout()
    dist_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_dist.png')
    os.makedirs(os.path.dirname(dist_plot_path), exist_ok=True)
    plt.savefig(dist_plot_path)

    # Plot 1b: 加速度层面等效扰动（用于和 STESO 的 X_hat 直接对比）
    plt.figure(figsize=(10, 12))
    w_units = ['m/s^2', 'deg/s^2', 'deg/s^2']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            w_real_plot = history['w_dist_real'][:, i]
            w_hat_plot = history['w_dist_hat'][:, i]
        else:
            w_real_plot = np.rad2deg(history['w_dist_real'][:, i])
            w_hat_plot = np.rad2deg(history['w_dist_hat'][:, i])
        plt.plot(history['t'], w_real_plot, 'k-', label='Real Equivalent Disturbance (H @ tau_ext)', linewidth=2, alpha=0.6)
        plt.plot(history['t'], w_hat_plot, 'r--', label='STESO Estimate (X_hat)', linewidth=1.5)
        plt.title(f'Equivalent Disturbance (Acceleration-level) - {titles[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Amplitude ({w_units[i]})')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    dist_w_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_equiv_w.png')
    plt.savefig(dist_w_plot_path)

    # Plot 1c: 总扰动（含 -C@qd - G）与 STESO 估计 X_hat 的对比
    plt.figure(figsize=(10, 12))
    total_w_units = ['m/s^2', 'deg/s^2', 'deg/s^2']
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            w_total_real_plot = history['w_total_real'][:, i]
            w_total_hat_plot = history['w_total_hat'][:, i]
        else:
            w_total_real_plot = np.rad2deg(history['w_total_real'][:, i])
            w_total_hat_plot = np.rad2deg(history['w_total_hat'][:, i])

        plt.plot(history['t'], w_total_real_plot, 'k-', label='Real Total Disturbance (H @ (tau_ext - C@qd - G))', linewidth=2, alpha=0.6)
        plt.plot(history['t'], w_total_hat_plot, 'r--', label='STESO Total Disturbance Estimate (X_hat)', linewidth=1.5)
        plt.title(f'Total Disturbance (Acceleration-level) - {titles[i]}')
        plt.xlabel('Time (s)')
        plt.ylabel(f'Amplitude ({total_w_units[i]})')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    total_w_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_total_equiv_w.png')
    plt.savefig(total_w_plot_path)

    # Plot 1d: 三条支链推力（用于观察饱和/分配）
    plt.figure(figsize=(10, 8))
    for k in range(3):
        plt.plot(history['t'], history['tau_legs'][:, k], linewidth=1.2, label=f'Leg {k+1} Force')
    plt.title('Leg Forces (tau_legs)')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    legs_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_tau_legs.png')
    plt.savefig(legs_plot_path)

    # Plot 1e: 腿长与腿速（用于观察腿运动学）
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    for k in range(3):
        plt.plot(history['t'], history['l_legs'][:, k], linewidth=1.2, label=f'Leg {k+1} Length')
    # （不再绘制 stroke/speed limit：已移除支链空间 PID 的约束统计配置）
    plt.title('Leg Lengths')
    plt.xlabel('Time (s)')
    plt.ylabel('Length (m)')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    for k in range(3):
        plt.plot(history['t'], history['ldot_legs'][:, k], linewidth=1.2, label=f'Leg {k+1} Speed')
    # （不再绘制 speed limit：已移除支链空间 PID 的约束统计配置）
    plt.title('Leg Speeds')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    legs_kin_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_leg_kinematics.png')
    plt.savefig(legs_kin_plot_path)

    # Plot 1f: 变负载曲线
    if enable_time_varying_payload and payload_profile != "none":
        plt.figure(figsize=(10, 4))
        plt.plot(history['t'], history['payload_m'], linewidth=2.0, label='m_platform(t)')
        plt.plot(history['t'], history['payload_dm'], linewidth=1.5, label='payload Δm(t)', alpha=0.8)
        plt.title('Time-varying Payload')
        plt.xlabel('Time (s)')
        plt.ylabel('Mass (kg)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        payload_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_payload.png')
        plt.savefig(payload_plot_path)
    else:
        payload_plot_path = None

    # Plot 1g: 重心偏置与等效力矩
    if enable_cg_bias and cg_bias_radius > 0.0:
        plt.figure(figsize=(10, 7))
        plt.subplot(2, 1, 1)
        plt.plot(history['t'], 1e3 * history['cg_xy'][:, 0], linewidth=1.8, label='x_cg (mm)')
        plt.plot(history['t'], 1e3 * history['cg_xy'][:, 1], linewidth=1.8, label='y_cg (mm)')
        plt.title('CG Bias (in-plane)')
        plt.xlabel('Time (s)')
        plt.ylabel('Offset (mm)')
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(history['t'], history['tau_cg'][:, 1], linewidth=1.8, label='tau_alpha_cg (Nm)')
        plt.plot(history['t'], history['tau_cg'][:, 2], linewidth=1.8, label='tau_beta_cg (Nm)')
        plt.title('Equivalent Gravity Torque from CG Bias')
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        cg_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_cg_bias.png')
        plt.savefig(cg_plot_path)
    else:
        cg_plot_path = None
    
    # Plot 2: System States (q) separate figure
    plt.figure(figsize=(10, 12))
    state_labels = ['Position Z (m)', 'Orientation Alpha (deg)', 'Orientation Beta (deg)']
    
    for i in range(3):
        plt.subplot(3, 1, i+1)
        if i == 0:
            q_plot = history['q'][:, i]
            q_ref_plot = np.ones_like(history['t']) * 0.5
        else:
            q_plot = np.rad2deg(history['q'][:, i])
            q_ref_plot = np.zeros_like(history['t'])

        plt.plot(history['t'], q_plot, 'b-', linewidth=1.5, label='Actual State')
        plt.plot(history['t'], q_ref_plot, 'g--', label='Target', alpha=0.7)
        plt.title(f'System State - {titles[i]}')
        plt.ylabel(state_labels[i])
        plt.xlabel('Time (s)')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    state_plot_path = os.path.join(os.path.dirname(__file__), '../outputs', 'steso_verification_states.png')
    plt.savefig(state_plot_path)
    
    print(
        f"验证结果已保存至:\n"
        f"  1. {dist_plot_path}\n"
        f"  2. {dist_w_plot_path}\n"
        f"  3. {total_w_plot_path}\n"
        f"  4. {legs_plot_path}\n"
        f"  5. {legs_kin_plot_path}\n"
        + (f"  6. {payload_plot_path}\n" if payload_plot_path else "")
        + (f"  7. {cg_plot_path}\n" if cg_plot_path else "")
        + f"  8. {state_plot_path}"
    )

if __name__ == "__main__":
    verify_steso()
