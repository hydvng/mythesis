# Chapter 4 工作计划：ESO + 硬切换 DRARL

## TL;DR

> **目标**: 在V5 Simplified基础上增加ESO扰动估计和硬切换机制，实现与Chapter3的对比实验
> 
> **核心方法**: SAC (V5纯ISE奖励) + ESO扰动补偿 + 误差阈值硬切换
> 
> **交付物**: 新环境、训练模型、控制图表、对比分析报告
> 
> **预计周期**: 1-2周

---

## Context

### 基础背景

- **Chapter 3**: 已完成，使用V5 Simplified (纯ISE奖励 + SAC)
- **V5结果**: ISE/step = 0.000011, 训练稳定，400 episodes
- **参考代码**: DRARLC-paper-code/phase3_drarl/ (ESO+硬切换实现)

### 核心需求

1. **保持V5训练方法**: 纯ISE奖励 + SAC算法 + 课程学习
2. **增加ESO**: 估计总扰动(海浪+模型不确定性)，输出作为控制补偿
3. **增加硬切换**: 基于误差阈值的中心区/边界区切换控制
4. **对比实验**: Chapter3 vs Chapter4性能对比

---

## Work Objectives

### 核心目标

1. 实现ESO模块，估计系统总扰动
2. 实现硬切换控制器（中心区/边界区）
3. 训练Chapter4模型
4. 生成对比图表和分析报告

### 交付物

- [ ] `env/rl_env_chapter4.py` - 集成ESO和硬切换的RL环境
- [ ] `env/eso_controller.py` - ESO控制器模块
- [ ] `experiments/train_chapter4.py` - 训练脚本
- [ ] `experiments/chapter4_models/` - 训练模型
- [ ] `figures/chapter4/` - 控制效果图
- [ ] `CHAPTER4_COMPARISON.md` - 对比分析报告

---

## Verification Strategy

### 测试策略

- **测试框架**: 沿用V5的测试脚本
- **对比指标**:
  - ISE/step (主要指标)
  - 控制力平滑度
  - 收敛时间
  - 约束满足率
- **测试场景**:
  - Sinusoidal轨迹
  - Constant轨迹  
  - 不同海浪条件 (Hs=1.0m, 2.0m)

### QA验证

- 每个任务完成后运行测试验证
- 对比实验确保公平（相同随机种子、相同测试场景）

---

## Execution Strategy

### Wave 1: 环境搭建 (并行进行)

```
├── T1: 创建chapter4目录结构
├── T2: 实现ESO控制器模块
├── T3: 实现硬切换控制器
└── T4: 创建Chapter4 RL环境
```

### Wave 2: 控制器整合

```
├── T5: 集成ESO到环境
├── T6: 集成硬切换逻辑
├── T7: 测试ESO+硬切换基本功能
└── T8: 调试与修正
```

### Wave 3: 训练

```
├── T9: 训练Chapter4模型 (300-400 episodes)
├── T10: 保存最佳模型
└── T11: 训练过程记录
```

### Wave 4: 对比与分析

```
├── T12: 生成Chapter4控制图
├── T13: 对比实验 (Chapter3 vs Chapter4)
├── T14: 性能分析报告
└── T15: 整理最终文档
```

---

## TODOs

### Wave 1: 环境搭建

- [ ] 1. 创建chapter4目录结构

  **What to do**:
  - 创建 env/, experiments/, figures/ 子目录
  - 复制V5环境作为基础模板

  **References**:
  - `chapter3/env/rl_env_v5_simplified.py` - V5环境模板

- [ ] 2. 实现ESO控制器模块

  **What to do**:
  - 实现二阶ESO: `ż = A*z + B*u + L*(y - z1)`
  - 估计扰动 d = f(x) + w(t)
  - 参考DRARLC论文的ESO设计

  **References**:
  - `DRARLC-paper-code/drarlc_impl/phase3_drarl/fixed_drarl_controller.py` - RBF估计参考

- [ ] 3. 实现硬切换控制器

  **What to do**:
  - 实现性能函数 ρ(t) = (ρ₀ - ρ∞)*exp(-κ*t) + ρ∞
  - 计算误差边界比值 |e|/ρ
  - 实现中心区/边界区切换逻辑

  **References**:
  - DRARLC论文: 中心区/边界区控制律

- [ ] 4. 创建Chapter4 RL环境

  **What to do**:
  - 基于rl_env_v5_simplified.py
  - 集成ESO模块
  - 集成硬切换控制器
  - 保持纯ISE奖励函数

  **References**:
  - `chapter3/env/rl_env_v5_simplified.py` - V5环境

### Wave 2: 控制器整合

- [ ] 5. 集成ESO到环境

  **What to do**:
  - 在step()中调用ESO更新
  - 将ESO估计值用于控制计算
  - 记录扰动估计历史

- [ ] 6. 集成硬切换逻辑

  **What to do**:
  - 计算性能函数 ρ(t)
  - 判断当前所在区域
  - 应用对应控制律

- [ ] 7. 测试ESO+硬切换基本功能

  **What to do**:
  - 运行单回合仿真测试
  - 验证ESO扰动估计收敛
  - 验证硬切换正常工作

- [ ] 8. 调试与修正

  **What to do**:
  - 修复发现的问题
  - 确保数值稳定性

### Wave 3: 训练

- [ ] 9. 训练Chapter4模型

  **What to do**:
  - 使用V5的训练配置
  - 300-400 episodes训练
  - 保存best_model.pt

  **References**:
  - `chapter3/experiments/v5_simplified/train_v5_simplified.py` - V5训练配置

  **QA Scenarios**:
  - 训练曲线单调下降
  - 无发散现象
  - ISE持续改善

- [ ] 10. 保存最佳模型

  **What to do**:
  - 记录最佳episode
  - 保存模型权重

- [ ] 11. 训练过程记录

  **What to do**:
  - 保存training_history.json
  - 记录奖励变化

### Wave 4: 对比与分析

- [ ] 12. 生成Chapter4控制图

  **What to do**:
  - 生成position, error, control子图
  - z单位mm, alpha/beta单位度

  **References**:
  - `chapter3/experiments/v5_simplified/plot_control_results.py` - V5绘图

- [ ] 13. 对比实验

  **What to do**:
  - 使用相同测试场景对比
  - Chapter3 V5 vs Chapter4
  - 记录ISE差异

- [ ] 14. 性能分析报告

  **What to do**:
  - 撰写CHAPTER4_COMPARISON.md
  - 总结ESO和硬切换的效果

- [ ] 15. 整理最终文档

  **What to do**:
  - 整理所有输出文件
  - 更新VERSION_SUMMARY.md

---

## Success Criteria

### 验收标准

- [ ] Chapter4模型训练完成，无发散
- [ ] ISE/step 优于或接近 Chapter3 V5
- [ ] 硬切换机制正常工作
- [ ] ESO扰动估计收敛
- [ ] 生成对比分析报告

### 性能对比目标

| 指标 | Chapter3 V5 | Chapter4 目标 |
|------|-------------|--------------|
| ISE/step | 0.000011 | ≤ 0.000011 |
| 训练稳定性 | 稳定 | 稳定 |
| 约束满足 | 100% | 100% |

---

*Generated: 2026-02-25*
