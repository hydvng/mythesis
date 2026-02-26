# V4 Improved变体完整训练方案

## ⚠️ 重要说明

之前的测试有严重缺陷：
- ❌ 只测试了inference（用已训练模型在不同环境测试）
- ❌ 没有真正训练V4 Improved模型
- ❌ 训练时间太短，看不出效果

## ✅ 正确的测试方案

需要为每个变体**从头训练完整模型**：

### 训练配置

每个变体需要：
- **Stage 1**: 20秒回合, 200 episodes, Hs=1.0m
- **Stage 2**: 30秒回合, 250 episodes, Hs=2.0m  
- **Stage 3**: 40秒回合, 300 episodes, Hs=2.0m

**预计时间**：
- 单个变体：3-4小时
- 全部3个变体：9-12小时

### 变体列表

1. **Original_V4_Improved**（原始改进版）
   - diverge_threshold = 0.5
   - warning_threshold = 0.4

2. **Variant1_Relaxed_Threshold**（放宽阈值）
   - diverge_threshold = 1.0
   - warning_threshold = 0.4

3. **Variant2_No_Warning_Penalty**（移除预警惩罚）
   - diverge_threshold = 0.5
   - warning_threshold = 0.4
   - 预警惩罚 = 0.0

## 运行方法

### 方案1：完整训练所有变体（推荐但耗时）

```bash
cd /home/haydn/Documents/AERAOFMINE/mythesis/simulation/chapter3/experiments/v4_comparison

# 使用leisaac环境运行
/home/haydn/miniconda3/envs/leisaac/bin/python train_all_variants.py
```

**预计时间**：9-12小时

### 方案2：只训练一个变体（快速测试）

创建单个变体训练脚本：

```bash
# 训练Original_V4_Improved
/home/haydn/miniconda3/envs/leisaac/bin/python -c "
import sys
sys.path.insert(0, 'train_all_variants.py')
from train_all_variants import V4ImprovedTrainer

trainer = V4ImprovedTrainer(
    'Original_V4_Improved',
    {'diverge_threshold': 0.5, 'warning_threshold': 0.4},
    './trained_variants'
)
trainer.train()
"
```

**预计时间**：3-4小时

### 方案3：后台运行（长时间任务）

```bash
# 使用nohup后台运行
nohup /home/haydn/miniconda3/envs/leisaac/bin/python train_all_variants.py > training.log 2>&1 &

# 查看进度
tail -f training.log
```

## 训练结果

训练完成后，每个变体会生成：

```
trained_variants/
├── Original_V4_Improved/
│   ├── Stage1_best.pt
│   ├── Stage1_final.pt
│   ├── Stage2_best.pt
│   ├── Stage2_final.pt
│   ├── Stage3_best.pt
│   ├── Stage3_final.pt
│   └── training_history.json
├── Variant1_Relaxed_Threshold/
│   └── ...
└── Variant2_No_Warning_Penalty/
    └── ...
```

## 训练后评估

训练完成后，运行完整评估：

```bash
/home/haydn/miniconda3/envs/leisaac/bin/python evaluate_trained_variants.py
```

## 快速开始（推荐）

如果想快速验证，建议先只训练**一个stage**：

```bash
# 修改train_all_variants.py，只保留Stage1
# 将max_episodes从200改为50
# 这样只需要约30分钟就能看到初步结果
```

## 监控训练

### 查看GPU使用情况
```bash
nvidia-smi -l 1
```

### 查看训练日志
```bash
# 实时查看
tail -f training.log

# 查看最近100行
tail -n 100 training.log
```

### 检查训练进度
训练脚本会每10个episode打印一次进度：
```
Episode  10 | Reward:   -45.32 | Avg(10):   -52.14 | Best: -45.32 (#0) | NoImp:  0 | Time: 2.3min
```

## 预期结果

根据之前的分析，预期结果：

1. **Original_V4_Improved** vs **Baseline_V4_Optimized**
   - 性能应该相近（差异<10%）
   - 因为diverge_threshold都是0.5

2. **Variant1_Relaxed_Threshold**（diverge_threshold=1.0）
   - 如果在训练时就使用1.0，可能看到：
     - 更稳定的训练（更少终止）
     - 但最终性能可能略差（因为允许更大的误差）

3. **Variant2_No_Warning_Penalty**
   - 可能比Original略好（少了额外的惩罚信号）

## 常见问题

### Q: 训练太慢怎么办？
A: 减少episodes数量（200→100, 250→125, 300→150）

### Q: 内存不足？
A: 减小batch_size（512→256）或replay_buffer容量

### Q: 想中断训练？
A: Ctrl+C，已保存的checkpoint不会丢失

### Q: 如何从checkpoint恢复？
A: 修改load_model参数指向对应的.pt文件

## 下一步

1. **短期**：先训练一个变体验证流程
2. **中期**：完成所有变体训练
3. **长期**：根据结果重新设计改进方案

---

**创建时间**：2026-02-24
**预计完成时间**：2026-02-25（如果今天开始训练）
