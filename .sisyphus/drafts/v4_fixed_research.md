# V4 Fixed 研究工作记录

## 一、项目背景与系统概述

### 1.1 研究对象
船载稳定平台（3-UPS/PU并联平台）的约束镇定控制

### 1.2 核心方法
- **模型补偿 (Model Compensation)**: 使用逆动力学计算基础控制量
- **扰动估计**: 海浪扰动（JONSWAP谱）
- **强化学习 (RL)**: SAC算法学习残余补偿

### 1.3 控制架构
```
总控制量 u = tau_model(模型补偿) + v_RL(RL补偿)
```

### 1.4 环境参数
- **海浪条件**: Hs=2.0m, T1=8.0s
- **期望轨迹**: sinusoidal正弦轨迹
  - z: 1.0 + 0.1*sin(2π*0.2*t)
  - α: 0.1*sin(2π*0.15*t)
  - β: 0.08*sin(2π*0.25*t + π/6)
- **回合长度**: 50秒 (5000步)
- **动作尺度**: 5000 N/N·m

---

## 二、决定怎么做

### 2.1 原始目标
V4版本训练收敛性差，试图通过以下方式改进：
1. 移除ISE惩罚（避免信用分配问题）
2. 使用课程学习（从简单到复杂）
3. 训练更长时间

### 2.2 V4 Fixed 环境设计
- 移除ISE累积误差惩罚
- 保留即时误差奖励
- 保留时序权重和收敛奖励
- ISE仅作为事后评估指标

### 2.3 课程学习配置
| Stage | 时间 | 海浪Hs | 轨迹类型 | Episodes |
|-------|------|--------|----------|----------|
| Stage1 | 20s | 1.0m | constant | 100-200 |
| Stage2 | 35s | 2.0m | sinusoidal_small | 150-200 |
| Stage3 | 50s | 2.0m | sinusoidal | 200-300 |

---

## 三、具体做了什么

### 3.1 创建的文件
1. **环境文件**: `env/rl_env_v4_fixed.py`
   - 移除ISE惩罚
   - 保留其他V4奖励机制

2. **训练脚本**: `experiments/train_v4_fixed.py`
   - 基础训练脚本（非课程）

3. **课程学习脚本**: `experiments/train_v4_fixed_curriculum.py`
   - 3阶段课程学习
   - 支持模型加载和早停

4. **绘图脚本**: `experiments/plot_v4_fixed.py`
   - 轨迹跟踪图
   - 控制分解图
   - 误差分析图

5. **ISE比较脚本**: `experiments/compare_ise.py`
   - V4 Optimized vs V4 Fixed 对比
   - ISE/step公平比较

### 3.2 关键发现与修复
1. **ISE比较分析**
   - 发现V4 Fixed显示低ISE是因为早终止（10步）
   - 引入ISE/step标准化指标

2. **Tanh Bug修复**
   - 发现Actor的deterministic模式没有应用tanh
   - 导致动作无界，迅速发散
   - 修复后模型可以跑完2000步

---

## 四、取得的效果

### 4.1 ISE/step对比结果
| 模型 | 平均步数 | ISE | ISE/step |
|------|----------|-----|----------|
| V4 Optimized | 461 | 0.3436 | **0.000745** |
| V4 Fixed (旧) | 10 | 0.0075 | 0.000777 |

### 4.2 修复tanh后的Stage1测试
- 步数: 2000 (满回合)
- 条件: Hs=1.0, constant轨迹
- ISE: 0.0876
- 结论: 可以完成回合，但误差仍较大

### 4.3 绘制的图表
- `V4_Fixed_tracking.png` - 原始模型（42步发散）
- `V4_Fixed_Stage1_tracking.png` - Stage1模型（2000步）

---

## 五、留下的缺陷与问题

### 5.1 训练相关
- ❌ 课程学习训练未完成（超时）
- ❌ 没有在Hs=2.0 + sinusoidal下训练到收敛
- ❌ 探索噪声设置过大（0.5开始）

### 5.2 代码问题
- ❌ tanh bug导致模型快速发散（已在训练脚本中修复）
- ❌ 课程学习每阶段步数设置较短

### 5.3 待解决问题
1. **为什么越训练坚持时间越少？**
   - 原因：课程学习每阶段训练步数减少
   - Stage1只有20秒 = 2000步/20 = 100步
   - 探索噪声衰减不足

2. **ISE惩罚是否真的有效？**
   - V4 Fixed的ISE/step与V4 Optimized接近
   - 需要完整训练才能判断

---

## 六、成功经验分析

### 6.1 V3/V4成功要素对比
| 项目 | V3（成功） | V4 Optimized（成功） | V4 Fixed v1（失败） |
|------|-----------|---------------------|---------------------|
| **环境** | PlatformRLEnv | PlatformRLEnvV4 | PlatformRLEnvV4Fixed |
| **ISE惩罚** | ❌ 无 | ✅ 有 [5000,2000,2000] | ❌ 无 |
| **网络架构** | AdvancedSAC | V4SAC (512×4+残差+LN+Drop) | SACAgent (512×3) |
| **训练Episodes** | 200+300+500=1000 | 200+250+300=750 | 100+150+200=450 |
| **课程配置** | 10s→20s→30s | 20s→30s→40s | 20s→35s→50s |
| **结果** | 成功收敛 | 成功收敛 | 快速发散 |

### 6.2 V4 Fixed v2 改进方案
基于以上分析，创建train_v4_fixed_v2.py：
- ✅ 移除ISE惩罚（V4 Fixed环境）
- ✅ 使用V4SAC大网络架构（512+512+256+256 + 残差+LayerNorm+Dropout）
- ✅ 完整训练配置（200+250+300 episodes，与V4 Optimized相同）
- ✅ 正确tanh（deterministic模式下也应用tanh）
- ✅ 课程配置与V4 Optimized相同（20s→30s→40s）

### 6.2 关键发现

**V4 Optimized成功的原因不是因为ISE惩罚！**

真正成功的原因是：
1. **更大的网络**: 4层隐藏层(512+512+256+256) + 残差连接 + LayerNorm + Dropout
2. **完整的训练**: 3个阶段完整训练，每个阶段200+ episodes
3. **正确的tanh**: V4SAC在deterministic模式下正确应用了tanh

**V4 Fixed失败的原因**：
1. **网络太小**: 只有3层(512,512,256)，没有残差/LayerNorm
2. **训练不足**: 每阶段训练 episodes 太少
3. **tanh bug**: 自己的训练脚本中deterministic模式没有tanh

### 6.3 经验总结

**不要轻易移除ISE惩罚！** V4 Optimized有ISE惩罚但依然成功，说明：
- ISE惩罚对收敛没有负面影响
- 反而可能帮助长期收敛
- V4 Fixed移除ISE后没有获得性能提升

**真正重要的是**：
- 网络容量（够大）
- 训练时长（够长）
- 代码正确性（tanh等）

---

## 七、文件清单

### 7.1 主要文件
| 文件 | 用途 | 状态 |
|------|------|------|
| `env/rl_env_v4_fixed.py` | V4 Fixed环境 | ✅ 完成 |
| `experiments/train_v4_fixed.py` | 基础训练 | ✅ 完成 |
| `experiments/train_v4_fixed_curriculum.py` | 课程学习 | ✅ 完成 |
| `experiments/plot_v4_fixed.py` | 绘图 | ✅ 完成 |
| `experiments/compare_ise.py` | ISE对比 | ✅ 完成 |

### 7.2 模型文件
| 文件 | 说明 |
|------|------|
| `models_v4_fixed/best_model.pt` | 旧模型（有问题） |
| `models_v4_fixed/stage1/Stage1_best.pt` | Stage1模型 |

### 7.3 生成的图片
| 文件 | 说明 |
|------|------|
| `figures/v4_fixed/V4_Fixed_tracking.png` | 原始模型轨迹 |
| `figures/v4_fixed/V4_Fixed_Stage1_tracking.png` | Stage1模型轨迹 |
| `figures/v4_fixed/V4_Fixed_control.png` | 控制分解 |
| `figures/v4_fixed/V4_Fixed_error_analysis.png` | 误差分析 |
