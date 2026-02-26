# mythesis

## 环境配置

### Conda环境

**训练环境**: `leisaac`

```bash
# 激活训练环境
conda activate leisaac
```

> **提示**: 使用 `leisaac` 环境进行强化学习训练，已预装 PyTorch、Gym 等依赖。

---

## 项目概述

毕业论文：**船载稳定平台的约束镇定控制研究**

核心方法：模型补偿 + ESO扰动估计 + 强化学习(RL) + 约束处理

---

## 工作进展

| 章节 | 状态 | 说明 |
|------|------|------|
| 第2章 | ✅ 完成 | 系统建模与约束分析 |
| 第3章 | 🔄 进行中 | 模型补偿+RL基础方法 |
| 第4章 | ⏳ 待开始 | ESO+硬切换DRARL |
| 第5章 | ⏳ 待开始 | 软过渡+自适应β DRARL |

### 详细进展

#### 第2章 ✅
- [x] 3-UPS/PU平台动力学模型
- [x] 海浪扰动模型 (JONSWAP谱)
- [x] 约束条件建模

#### 第3章 🔄
- [x] RL环境封装 (`env/rl_env.py`)
- [x] SAC算法实现 (`agents/sac_agent.py`)
- [x] 仿真脚本框架
- [ ] 完整训练与图表生成

---

## 目录结构

```
simulation/
├── common/                  # 公共模块
│   ├── platform_dynamics.py # 动力学模型
│   └── wave_disturbance.py  # 海浪扰动
├── chapter2/                # 第2章仿真
├── chapter3/                # 第3章：模型补偿+RL
├── chapter4/                # 第4章：ESO+硬切换
└── chapter5/                # 第5章：软过渡+自适应β
```

---

## 快速开始

```bash
# 激活环境
conda activate leisaac

# 运行第3章仿真
cd simulation/chapter3/experiments
python chapter3_simulation.py
```