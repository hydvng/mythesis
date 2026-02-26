#!/usr/bin/env python3
"""Check Roll RAO values at 180 degrees"""
import numpy as np
from scipy.io import loadmat
import sys

print("="*70)
print("CRITICAL CHECK: Roll RAO at 180 degrees")
print("="*70)

# Load MSS data
mat_path = '/home/haydn/Documents/AERAOFMINE/mythesis/simulation/disturbance/data/supply.mat'
data = loadmat(mat_path)
vessel = data['vessel']

motionRAO = vessel['motionRAO'][0, 0]
rao_freqs = motionRAO['w'][0, 0].flatten()
rao_headings = vessel['headings'][0, 0].flatten()

print("\n1. MSS Raw Data - Roll RAO (DOF 3):")
print("-" * 70)

amp_data = motionRAO['amp'][0, 0]
roll_rao = amp_data[0, 3][:, :, 0]

print(f"Roll RAO shape: {roll_rao.shape}")
print(f"Headings (deg): {np.degrees(rao_headings)}")

print("\nRoll RAO at each heading (averaged over frequency):")
for i, h in enumerate(rao_headings):
    h_deg = np.degrees(h)
    avg_val = np.mean(roll_rao[:, i])
    print(f"  {h_deg:3.0f}°: {avg_val:.8f}")

print("\n" + "="*70)
print("2. Python Interpolated RAO (wave_heading=180°):")
print("-" * 70)

sys.path.insert(0, '/home/haydn/Documents/AERAOFMINE/mythesis/simulation')
from disturbance.wave_disturbance import WaveDisturbance

wave = WaveDisturbance(
    Hs=2.0, T1=8.0,
    vessel_file='supply.mat',
    wave_heading=180.0,
    random_seed=42
)

print(f"\nInterpolated Roll RAO (first 10 frequencies):")
for i in range(min(10, len(wave.omegas))):
    omega = wave.omegas[i]
    roll = wave.RAO['M_alpha'][i]
    print(f"  ω={omega:.3f}: Roll={roll:.8f}")

print(f"\nMean Roll RAO: {np.mean(wave.RAO['M_alpha']):.8f}")
print(f"Max Roll RAO:  {np.max(wave.RAO['M_alpha']):.8f}")

print("\n" + "="*70)
print("3. Generate Disturbance and Check:")
print("-" * 70)

t = np.linspace(0, 100, 10000)
dist = wave.generate_disturbance(t)

print(f"\nDisturbance Statistics:")
print(f"  Fz std:      {np.std(dist[:,0]):.2f} N")
print(f"  M_alpha std: {np.std(dist[:,1]):.6f} N.m (SHOULD BE ~0!)")
print(f"  M_beta std:  {np.std(dist[:,2]):.4f} N.m")

if np.std(dist[:,1]) > 0.1:
    print("\n" + "!"*70)
    print("WARNING: Roll disturbance is NOT near zero!")
    print("Expected: ~0 for head sea (180°)")
    print("Actual:", np.std(dist[:,1]))
    print("!"*70)
else:
    print("\n✓ Roll disturbance is near zero (correct)")
