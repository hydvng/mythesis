import numpy as np

class STESO:
    """
    Super-Twisting Extended State Observer (STESO)
    
    用于估计3-DOF并联平台的总扰动（包含未建模动力学和外部扰动）。
    由用户提供的公式实现。
    
    Attributes:
        dim (int): 状态维度 (3)
        lambda1 (float): 滑模面参数 S = lambda1*e + e_dot
        beta1 (float): 观测器增益1 (主导S收敛)
        beta2 (float): 观测器增益2 (主导X收敛)
        dt (float): 采样时间
    """
    
    def __init__(self, 
                 dim: int = 3, 
                 lambda1: float = 5.0, 
                 beta1: float = 20.0, 
                 beta2: float = 10.0, 
                 dt: float = 0.002):
        self.dim = dim
        self.lambda1 = lambda1
        self.beta1 = beta1
        self.beta2 = beta2
        self.dt = dt
        
        # 估计状态
        self.S_hat = np.zeros(dim)
        self.X_hat = np.zeros(dim)
        
        self.is_initialized = False

    def init_state(self,
                   q: np.ndarray,
                   qd: np.ndarray,
                   q_des: np.ndarray = None,
                   qd_des: np.ndarray = None,
                   X_hat0: np.ndarray | None = None):
        """初始化观测器状态，避免启动时的巨大误差。

        Args:
            q: 初始测量位姿
            qd: 初始测量速度
            q_des: 初始期望位姿
            qd_des: 初始期望速度
            X_hat0: 扰动估计初值(\hat X(0))，形状为 [dim]；None 表示默认全 0
        """
        if q_des is None: q_des = np.zeros(self.dim)
        if qd_des is None: qd_des = np.zeros(self.dim)
        
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed
        
        self.S_hat = S.copy()
        # 扰动估计初值：默认 0，也可由外部指定
        if X_hat0 is None:
            self.X_hat = np.zeros(self.dim)
        else:
            X_hat0 = np.asarray(X_hat0, dtype=float).reshape(-1)
            if X_hat0.shape[0] != self.dim:
                raise ValueError(f"X_hat0 shape must be ({self.dim},), got {X_hat0.shape}")
            self.X_hat = X_hat0.copy()
        self.is_initialized = True

    def update(self, 
               q: np.ndarray, 
               qd: np.ndarray, 
               tau: np.ndarray, 
               H: np.ndarray, 
               q_des: np.ndarray = None, 
               qd_des: np.ndarray = None) -> np.ndarray:
        """
        更新观测器状态
        
        Args:
            q: 当前位姿 [3]
            qd: 当前速度 [3]
            tau: 控制力矩 [3]
            H: 惯性矩阵的逆 (M^-1) [3x3]
            q_des: 期望位姿 [3]
            qd_des: 期望速度 [3]
            
        Returns:
            X_hat: 总扰动估计值 [3]
        """
        # 默认期望为0
        if q_des is None: q_des = np.zeros(self.dim)
        if qd_des is None: qd_des = np.zeros(self.dim)
        
        # 自动初始化
        if not self.is_initialized:
            self.init_state(q, qd, q_des, qd_des)
        
        # 1. 计算真实滑模面 S
        # S = lambda1 * e + e_dot
        # e = q - q_des, e_dot = qd - qd_des
        e = q - q_des
        ed = qd - qd_des
        S = self.lambda1 * e + ed
        
        # 2. 计算估计误差 S_tilde
        S_tilde = self.S_hat - S 
        
        # 3. 计算非线性项
        # Sig^1/2(S_tilde) = |S_tilde|^0.5 * sign(S_tilde)
        # 为防止数值问题，S_tilde绝对值极小时可直接视为0
        abs_S_tilde = np.abs(S_tilde)
        sig_half = np.sqrt(abs_S_tilde) * np.sign(S_tilde)
        
        # Sig(S_tilde) = sign(S_tilde)
        sig_one = np.sign(S_tilde)
        
        # 4. 已知动态项 F
        # 根据推导: F = lambda1 * q_dot 
        F = self.lambda1 * qd
        
        # 5. RK4 积分更新
        # 定义状态导数函数
        # dot_S_hat = H*tau + F + X_hat + beta1 * Sig^1/2(S - S_hat)
        # dot_X_hat = beta2 * Sig(S - S_hat)
        
        V = (H @ tau) + F  # 已知输入项 (假设在 dt 内为常数)
        
        def dynamics(S_hat_val, X_hat_val):
            # 计算误差 S_tilde = S (测量值) - S_hat (当前估计)
            # 注意: S 在这里取当前时刻测量值 (Zero-Order Hold)
            err = S - S_hat_val
            
            abs_err = np.abs(err)
            sig_half = np.sqrt(abs_err) * np.sign(err)
            sig_one = np.sign(err)
            
            dS = V + X_hat_val + self.beta1 * sig_half
            dX = self.beta2 * sig_one
            return dS, dX
            
        # k1
        k1_S, k1_X = dynamics(self.S_hat, self.X_hat)
        
        # k2
        k2_S, k2_X = dynamics(self.S_hat + 0.5 * self.dt * k1_S, 
                              self.X_hat + 0.5 * self.dt * k1_X)
                              
        # k3
        k3_S, k3_X = dynamics(self.S_hat + 0.5 * self.dt * k2_S,
                              self.X_hat + 0.5 * self.dt * k2_X)
                              
        # k4
        k4_S, k4_X = dynamics(self.S_hat + self.dt * k3_S,
                              self.X_hat + self.dt * k3_X)
        
        # Update
        self.S_hat += (self.dt / 6.0) * (k1_S + 2*k2_S + 2*k3_S + k4_S)
        self.X_hat += (self.dt / 6.0) * (k1_X + 2*k2_X + 2*k3_X + k4_X)
        
        return self.X_hat.copy()

    def calculate_convergence_upper_bound(self, S: np.ndarray, S_hat: np.ndarray, 
                                          X_true: np.ndarray, X_hat: np.ndarray, 
                                          delta: float) -> tuple[float, bool, float]:
        """
        根据 Lyapunov 理论计算 STESO 的收敛时间上界 t_s1
        
        Args:
            S (np.ndarray): 真实滑模面值
            S_hat (np.ndarray): 估计滑模面值
            X_true (np.ndarray): 真实总扰动 (仅仿真可用)
            X_hat (np.ndarray): 估计总扰动
            delta (float): 扰动导数上界 (满足 |X_dot| <= delta)
            
        Returns:
            t_s1 (float): 收敛时间上界 (秒)
            is_stable (bool): 参数是否满足收敛性条件 (Q正定)
            gamma (float): 收敛速率参数 gamma
        """
        # 1. 构造误差向量 eta components
        # S_tilde = S - S_hat, X_tilde = X_true - X_hat
        S_tilde = S - S_hat
        X_tilde = X_true - X_hat
        
        # eta1 = |S_tilde|^0.5 * sign(S_tilde)
        eta1 = np.sqrt(np.abs(S_tilde)) * np.sign(S_tilde)
        # eta2 = X_tilde
        eta2 = X_tilde
        
        # 2. 构造 P1 矩阵 (2x2 Block Coefficients)
        # P1 = 0.5 * [[ (4*b2 + b1^2), -b1 ],
        #             [ -b1,            2  ]]
        p11 = 0.5 * (4 * self.beta2 + self.beta1**2)
        p12 = 0.5 * (-self.beta1)
        p22 = 0.5 * 2.0
        
        P_block = np.array([[p11, p12],
                            [p12, p22]])
                            
        # 3. 构造 Q1 矩阵 (2x2 Block Coefficients)
        # Q1 = [[ b1*b2 + 0.5*b1^3 - delta*(b2+1),  -0.5*b1^2 ],
        #       [ -0.5*b1^2,                        0.5*b1 - delta ]]
        q11 = self.beta1 * self.beta2 + 0.5 * self.beta1**3 - delta * (self.beta2 + 1)
        q12 = -0.5 * self.beta1**2
        q22 = 0.5 * self.beta1 - delta
        
        Q_block = np.array([[q11, q12],
                            [q12, q22]])
        print("Q_block:\n", Q_block)
        # 4. 计算特征值
        eig_P = np.linalg.eigvalsh(P_block)
        eig_Q = np.linalg.eigvalsh(Q_block)
        print("Eigenvalues of Q_block:", eig_Q)
        print("Eigenvalues of P_block:", eig_P)
        
        lambda_min_P = np.min(eig_P)
        lambda_max_P = np.max(eig_P)
        lambda_min_Q = np.min(eig_Q)
        
        # 检查稳定性条件: Q1 必须正定 => lambda_min_Q > 0
        is_stable = lambda_min_Q > 0
        
        if not is_stable:
            return float('inf'), False, 0.0
            
        # 5. 计算 Gamma
        # 理论说明: 
        # 标准有限时间稳定性条件为: V_dot <= -gamma * V^0.5
        # 结合 V <= lambda_max(P) * ||eta||^2 和 V_dot <= -lambda_min(Q) * ||eta||^2 * lambda_min(P)^0.5 / V^0.5
        # 推导可得: gamma = (lambda_min(Q) * lambda_min(P)^0.5) / lambda_max(P)
        # 此 gamma 值对应收敛时间公式: t_s1 <= (2/gamma) * V(0)^0.5
        gamma = (lambda_min_Q * np.sqrt(lambda_min_P)) / lambda_max_P
        
        # 6. 计算 Lyapunov 函数值 V(0)
        # V = eta^T * P1 * eta
        # 由于 P1 是分块对角阵 (Kron(P_block, I)), V 是每个轴的 V_i 之和
        # V = sum( [eta1_i, eta2_i] @ P_block @ [eta1_i, eta2_i].T )
        V = 0.0
        for i in range(self.dim):
            vec = np.array([eta1[i], eta2[i]])
            V += vec @ P_block @ vec
            
        # 7. 计算时间上界
        # t_s1 <= (2 / gamma) * sqrt(V)
        # 防止 V < 0 数值误差
        V = max(0.0, V)
        t_s1 = (2.0 / gamma) * np.sqrt(V)
        
        return t_s1, is_stable, gamma
    
    @staticmethod
    def suggest_parameters(delta: float, 
                           t_desired: float, 
                           initial_S_err: float = 0.5, 
                           initial_X_err: float = 10.0,
                           beta1_range: tuple = (50, 2000),
                           beta2_scale_range: tuple = (1.0, 50.0)) -> tuple[float, float, float]:
        """
        根据扰动界 delta 和期望收敛时间 t_desired，搜索满足条件的最小参数组合。
        注意：这是一个启发式搜索，返回的是第一组满足条件的"较优"参数。
        
        Args:
            delta: 扰动导数上界
            t_desired: 期望收敛时间 (秒)
            initial_S_err: 预估的滑模面初始误差范数 (||S(0)-S_hat(0)||)
            initial_X_err: 预估的扰动初始误差范数 (||X(0)-X_hat(0)||)
            beta1_range: beta1 搜索范围 (min, max)
            beta2_scale_range: beta2 相对于 (delta*beta1) 的倍数范围
            
        Returns:
            best_beta1, best_beta2, calculated_time
        """
        best_beta1 = None
        best_beta2 = None
        min_cost = float('inf') # Cost = beta1 + w*beta2 (prefer smaller gains)
        found_time = float('inf')
        
        # 构造初始误差向量范数用于估算 V(0)
        # 简化假设: 单轴误差，或者这已经是总范数
        # eta1 = |S|^0.5 * sign(S), norm(eta1)^2 = sum(|S|)
        # 在这里我们近似处理：假设 V 差不多由 norm 决定
        # 更严谨的做法需要具体的 P 矩阵，我们在循环里算
        
        # 粗略网格搜索
        # Step 1: Beta1 
        # 约束: beta1 > 2*delta
        start_b1 = max(beta1_range[0], 2.1 * delta)
        b1_candidates = np.linspace(start_b1, beta1_range[1], 20)
        
        for b1 in b1_candidates:
            # Step 2: Beta2
            # 约束: beta2 > delta * beta1 (近似) / Determinant check
            # 搜索 beta2 = k * (delta * b1), k in scale_range (e.g., 0.8 to 5.0)
            # 因为 condition 2 大概是 beta2 * b1 > delta^2 (or similar), 实际上 beta2 需要比较大
            # 根据之前验证，beta2 > delta * b1 是强约束
            
            # 我们直接在 log 空间搜索 beta2，起点稍微放宽一点
            min_b2_est = delta * b1 * 0.8
            max_b2_est = delta * b1 * beta2_scale_range[1]
            b2_candidates = np.geomspace(max(1.0, min_b2_est), max_b2_est, 15)
            
            for b2 in b2_candidates:
                # 1. 快速检查稳定性 (Determinant)
                q11 = b1 * b2 + 0.5 * b1**3 - delta * (b2 + 1)
                q12 = -0.5 * b1**2
                q22 = 0.5 * b1 - delta
                
                det_Q = q11 * q22 - q12**2
                if det_Q <= 0 or q22 <= 0:
                    continue
                    
                # 2. 计算收敛时间
                # Reconstruct needed matrices eigenvalues
                # P1
                p11 = 0.5 * (4 * b2 + b1**2)
                p12 = 0.5 * (-b1)
                p22 = 1.0 # 0.5 * 2
                
                # Eigenvalues of P (2x2)
                # trace = p11 + p22
                # det = p11*p22 - p12^2
                tr_P = p11 + p22
                det_P = p11 * p22 - p12**2
                gap_P = np.sqrt(tr_P**2 - 4*det_P)
                eig_P_min = (tr_P - gap_P) / 2
                eig_P_max = (tr_P + gap_P) / 2
                
                # Eigenvalues of Q (2x2)
                tr_Q = q11 + q22
                # det_Q already calc
                gap_Q = np.sqrt(tr_Q**2 - 4*det_Q)
                eig_Q_min = (tr_Q - gap_Q) / 2
                
                gamma = (eig_Q_min * np.sqrt(eig_P_min)) / eig_P_max
                
                # Estimate V(0)
                # eta = [ |Stilde|^0.5, Xtilde ]
                # 假设 Stilde = initial_S_err (scalar approximation), Xtilde = initial_X_err
                # 这是一个保守估计/标量估计
                eta1_val = np.sqrt(initial_S_err) 
                eta2_val = initial_X_err
                
                # Simplified V_scalar = [eta1, eta2] * P * [eta1, eta2]^T
                vec = np.array([eta1_val, eta2_val])
                # Manual vec @ P @ vec
                V_est = vec[0]*(p11*vec[0] + p12*vec[1]) + vec[1]*(p12*vec[0] + p22*vec[1])
                
                t_calc = (2.0 / gamma) * np.sqrt(V_est)
                
                if t_calc <= t_desired:
                    # Found a valid pair
                    # Calculate cost (minimize total gain magnitude)
                    cost = b1 + 0.05 * b2 
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_beta1 = b1
                        best_beta2 = b2
                        found_time = t_calc
        
        return best_beta1, best_beta2, found_time
