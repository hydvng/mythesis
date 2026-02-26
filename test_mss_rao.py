#!/usr/bin/env python3
"""验证MSS RAO扰动模型"""
import numpy as np
import sys
sys.path.insert(0, '/home/haydn/Documents/AERAOFMINE/mythesis/simulation/common')

from wave_disturbance import WaveDisturbance

print("="*70)
print("验证新的MSS RAO扰动模型")
print("="*70)

# 测试1: Supply船型，海况4
print("\n【测试1】Supply船型，海况4 (Hs=2.0m, T1=8.0s)")
wave1 = WaveDisturbance(
    Hs=2.0, T1=8.0,
    vessel_file='supply.mat',
    wave_heading=180.0,
    n_components=50,
    random_seed=42
)

t = np.linspace(0, 200, 20000)
dist1 = wave1.generate_disturbance(t)

print(f"  Fz:      mean={np.mean(dist1[:,0]):.2f}N, std={np.std(dist1[:,0]):.2f}N")
print(f"  M_alpha: std={np.std(dist1[:,1]):.4f}N·m")
print(f"  M_beta:  std={np.std(dist1[:,2]):.4f}N·m")

# 检查物理合理性
m_platform = 34.754
g = 9.81
Fz_std = np.std(dist1[:,0])
print(f"\n  扰动标准差: {Fz_std:.2f}N = {Fz_std/(m_platform*g)*100:.2f}% 平台重力")

# 测试2: 不同浪向角
print("\n【测试2】不同浪向角的影响")
for heading in [0, 90, 180]:
    wave_h = WaveDisturbance(
        Hs=2.0, T1=8.0,
        vessel_file='supply.mat',
        wave_heading=heading,
        n_components=50,
        random_seed=42
    )
    dist_h = wave_h.generate_disturbance(t[:5000])
    print(f"  浪向{heading:3d}°: Fz_std={np.std(dist_h[:,0]):6.2f}N, "
          f"M_roll_std={np.std(dist_h[:,1]):6.4f}N·m, "
          f"M_pitch_std={np.std(dist_h[:,2]):6.4f}N·m")

# 测试3: S175船型
print("\n【测试3】S175集装箱船")
wave2 = WaveDisturbance(
    Hs=2.0, T1=8.0,
    vessel_file='s175.mat',
    wave_heading=180.0,
    n_components=50,
    random_seed=42
)

dist2 = wave2.generate_disturbance(t)
print(f"  Fz:      std={np.std(dist2[:,0]):.2f}N")
print(f"  M_alpha: std={np.std(dist2[:,1]):.4f}N·m")
print(f"  M_beta:  std={np.std(dist2[:,2]):.4f}N·m")

# 测试4: 不同海况等级
print("\n【测试4】不同海况等级 (Supply船)")
for sea_state in [2, 4, 6]:
    params = wave1.SEA_STATE_TABLE[sea_state]
    wave_ss = WaveDisturbance(
        Hs=params['Hs'], T1=params['T1'],
        vessel_file='supply.mat',
        wave_heading=180.0,
        n_components=50,
        random_seed=42
    )
    dist_ss = wave_ss.generate_disturbance(t[:5000])
    Fz_percent = np.std(dist_ss[:,0]) / (m_platform * g) * 100
    print(f"  海况{sea_state} (Hs={params['Hs']}m): "
          f"Fz_std={np.std(dist_ss[:,0]):6.2f}N ({Fz_percent:5.2f}%重力)")

# 测试5: RAO曲线检查
print("\n【测试5】RAO曲线峰值检查")
rao = wave1.get_rao_curve()
print(f"  Heave RAO峰值: {np.max(rao['Heave']):.4f} @ {rao['freq'][np.argmax(rao['Heave'])]:.2f} rad/s")
print(f"  Roll RAO峰值:  {np.max(rao['Roll']):.4f} rad/m")
print(f"  Pitch RAO峰值: {np.max(rao['Pitch']):.4f} rad/m")

# 测试6: 重现性检查
print("\n【测试6】随机种子重现性")
wave_a = WaveDisturbance(Hs=2.0, T1=8.0, vessel_file='supply.mat', 
                         wave_heading=180.0, random_seed=123)
wave_b = WaveDisturbance(Hs=2.0, T1=8.0, vessel_file='supply.mat', 
                         wave_heading=180.0, random_seed=123)
dist_a = wave_a.generate_disturbance(np.linspace(0, 10, 100))
dist_b = wave_b.generate_disturbance(np.linspace(0, 10, 100))
print(f"  相同种子生成结果差异: {np.max(np.abs(dist_a - dist_b)):.10f} (应≈0)")

print("\n" + "="*70)
print("验证完成!")
print("="*70)
