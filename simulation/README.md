# 毕业论文仿真代码目录结构

## 目录说明

```
simulation/
├── common/                          # 公共模块（各章节共享）
│   ├── __init__.py
│   ├── platform_dynamics.py        # 3-UPS/PU平台动力学模型
│   └── wave_disturbance.py         # 海浪扰动模型（JONSWAP谱）
│
├── chapter2/                        # 第2章：系统建模与约束分析
│   ├── experiments/
│   │   └── chapter2_simulation.py  # 生成图2-1到图2-5
│   └── figures/                     # 第2章生成的图片
│
├── chapter3/                        # 第3章：基于模型补偿与RL的约束镇定控制
│   ├── env/                         # RL环境
│   │   └── rl_env.py               # 强化学习环境封装
│   ├── agents/                      # RL智能体
│   │   └── sac_agent.py            # SAC算法实现
│   ├── experiments/                 # 实验脚本
│   └── figures/                     # 生成的图片
│
├── chapter4/                        # 第4章：基于ESO的DRARL镇定控制
│   ├── experiments/
│   └── figures/
│
└── chapter5/                        # 第5章：基于软过渡的DRARL镇定控制
    ├── experiments/
    └── figures/
```

## 使用说明

### 第2章仿真

```bash
cd simulation
python3 chapter2/experiments/chapter2_simulation.py
```

生成图片保存在：`chapter2/figures/`

### 导入公共模块

在任何章节的代码中，使用以下方式导入公共模块：

```python
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'common'))

from platform_dynamics import ParallelPlatform3DOF
from wave_disturbance import WaveDisturbance
```

## 模块说明

### platform_dynamics.py

3-UPS/PU并联平台完整动力学模型，包含：
- 运动学计算（正逆运动学、雅可比矩阵）
- 动力学计算（质量矩阵、科里奥利矩阵、重力、摩擦力）
- 正逆动力学求解
- 能量计算
- 约束检查
- 数值积分

### wave_disturbance.py

海浪扰动模型，包含：
- JONSWAP谱生成
- 多频率分量叠加
- 三自由度扰动（升沉、横摇、纵摇）

## 章节对应关系

| 章节 | 目录 | 主要内容 |
|------|------|----------|
| 第2章 | `chapter2/` | 系统建模与约束分析验证 |
| 第3章 | `chapter3/` | 模型补偿+RL基础方法 |
| 第4章 | `chapter4/` | ESO+硬切换DRARL |
| 第5章 | `chapter5/` | 软过渡+自适应β DRARL |

## 注意事项

1. 所有章节的代码都依赖`common/`目录中的模块
2. 各章节生成的图片保存在对应`figures/`目录
3. 不要跨章节引用代码，如有公共需求请添加到`common/`
