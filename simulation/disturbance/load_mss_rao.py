#!/usr/bin/env python3
"""
读取MSS运动RAO数据并转换为Python可用的格式
"""

from scipy.io import loadmat
import numpy as np
import json

def load_mss_motion_rao(mat_file_path):
    """
    从MSS的.mat文件加载运动RAO数据
    
    Args:
        mat_file_path: .mat文件路径
        
    Returns:
        dict: 包含运动RAO数据的字典
    """
    # 加载MAT文件
    data = loadmat(mat_file_path)
    
    # 提取vessel结构体
    vessel = data['vessel']
    
    # 提取运动RAO数据
    motionRAO = vessel['motionRAO'][0, 0]
    forceRAO = vessel['forceRAO'][0, 0]
    
    # 提取频率
    w_motion = motionRAO['w'][0, 0].flatten()  # 运动RAO频率 [rad/s]
    w_force = forceRAO['w'][0, 0].flatten()    # 力RAO频率 [rad/s]
    
    # 提取浪向角
    headings = vessel['headings'][0, 0].flatten()  # [rad]
    
    # 提取速度
    velocities = vessel['velocities'][0, 0].flatten()  # [m/s]
    
    # 提取船体主要参数
    main = vessel['main'][0, 0]
    ship_params = {
        'name': str(main['name'][0, 0][0]),
        'Lpp': float(main['Lpp'][0, 0]),
        'B': float(main['B'][0, 0]),
        'T': float(main['T'][0, 0]),
        'm': float(main['m'][0, 0]),
        'GM_T': float(main['GM_T'][0, 0]),  # 横向稳心高度
        'GM_L': float(main['GM_L'][0, 0]),  # 纵向稳心高度
    }
    
    # 提取6自由度运动RAO幅值和相位
    # 注意：数据结构是 (1, 6)，每个元素是 (频率, 方向, 速度) 的数组
    rao_motion_amp = []
    rao_motion_phase = []
    
    amp_data = motionRAO['amp'][0, 0]
    phase_data = motionRAO['phase'][0, 0]
    
    for dof in range(6):
        # 运动RAO: (频率, 方向, 速度)
        amp = amp_data[0, dof]  # 形状: (频率, 方向, 速度)
        phase = phase_data[0, dof]
        rao_motion_amp.append(amp)
        rao_motion_phase.append(phase)
    
    # 提取6自由度力RAO幅值和相位
    rao_force_amp = []
    rao_force_phase = []
    
    f_amp_data = forceRAO['amp'][0, 0]
    f_phase_data = forceRAO['phase'][0, 0]
    
    for dof in range(6):
        # 力RAO: (频率, 方向, 速度)
        amp = f_amp_data[0, dof]
        phase = f_phase_data[0, dof]
        rao_force_amp.append(amp)
        rao_force_phase.append(phase)
    
    result = {
        'ship_params': ship_params,
        'motion': {
            'freq': w_motion,           # [rad/s]
            'headings': headings,        # [rad], 0=following, pi=head
            'velocities': velocities,    # [m/s]
            'amp': rao_motion_amp,       # 6个DOF的幅值列表
            'phase': rao_motion_phase,   # 6个DOF的相位列表
        },
        'force': {
            'freq': w_force,
            'headings': headings,
            'velocities': velocities,
            'amp': rao_force_amp,
            'phase': rao_force_phase,
        }
    }
    
    return result


def print_rao_info(rao_data):
    """打印RAO数据信息"""
    ship = rao_data['ship_params']
    motion = rao_data['motion']
    
    print("=" * 70)
    print(f"船型: {ship['name']}")
    print("=" * 70)
    print(f"\n船体参数:")
    print(f"  垂线间长 Lpp: {ship['Lpp']:.2f} m")
    print(f"  船宽 B: {ship['B']:.2f} m")
    print(f"  吃水 T: {ship['T']:.2f} m")
    print(f"  排水量 m: {ship['m']:.2f} kg")
    print(f"  横稳心高 GM_T: {ship['GM_T']:.2f} m")
    print(f"  纵稳心高 GM_L: {ship['GM_L']:.2f} m")
    
    print(f"\n运动RAO数据:")
    print(f"  频率范围: {motion['freq'][0]:.3f} ~ {motion['freq'][-1]:.3f} rad/s")
    print(f"  频率点数: {len(motion['freq'])}")
    print(f"  浪向角: {np.degrees(motion['headings'])} degrees")
    print(f"  浪向角点数: {len(motion['headings'])}")
    print(f"  航速: {motion['velocities']} m/s")
    
    print(f"\n各自由度运动RAO幅值范围:")
    dof_names = ['Surge (m/m)', 'Sway (m/m)', 'Heave (m/m)', 
                 'Roll (rad/m)', 'Pitch (rad/m)', 'Yaw (rad/m)']
    for dof, name in enumerate(dof_names):
        amp = motion['amp'][dof]
        print(f"  {name}: {amp.min():.4f} ~ {amp.max():.4f}")


def interpolate_rao(rao_data, freq_target, heading_target, vel_idx=0):
    """
    在指定频率和浪向角处插值RAO
    
    Args:
        rao_data: load_mss_motion_rao的返回值
        freq_target: 目标频率 [rad/s]
        heading_target: 目标浪向角 [rad]
        vel_idx: 速度索引 (0=静止)
        
    Returns:
        ndarray: 6个DOF的RAO幅值
    """
    from scipy.interpolate import RegularGridInterpolator, interp1d
    
    motion = rao_data['motion']
    freqs = motion['freq']
    headings = motion['headings']
    
    # 6 DOF结果
    rao_interp = np.zeros(6)
    
    for dof in range(6):
        # 获取该DOF的RAO表 (频率, 方向)
        amp_table = motion['amp'][dof][:, :, vel_idx]
        
        # 2D插值 - 使用RegularGridInterpolator
        # 注意：需要处理角度环绕 (0-360度)
        f = RegularGridInterpolator(
            (freqs, headings), 
            amp_table,
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # 查询点
        rao_interp[dof] = f([[freq_target, heading_target]])[0]
    
    return rao_interp


if __name__ == '__main__':
    # 可用的MSS船型文件
    vessel_files = {
        'supply': '/home/haydn/Documents/AERAOFMINE/MSS/HYDRO/vessels_shipx/supply/supply.mat',
        's175': '/home/haydn/Documents/AERAOFMINE/MSS/HYDRO/vessels_shipx/s175/s175.mat',
        'tanker': '/home/haydn/Documents/AERAOFMINE/MSS/HYDRO/vessels_wamit/tanker/tanker.mat',
        'semisub': '/home/haydn/Documents/AERAOFMINE/MSS/HYDRO/vessels_wamit/semisub/semisub.mat',
    }
    
    # 加载并显示supply船的RAO数据
    print("\n正在加载 Supply Vessel 的RAO数据...")
    rao_supply = load_mss_motion_rao(vessel_files['supply'])
    print_rao_info(rao_supply)
    
    # 示例：在特定频率和浪向角处插值RAO
    print("\n" + "=" * 70)
    print("示例：在 ω=1.0 rad/s, 迎浪(180°) 处插值RAO")
    print("=" * 70)
    
    freq = 1.0  # rad/s
    heading = np.pi  # 180 degrees, head sea
    
    rao_values = interpolate_rao(rao_supply, freq, heading)
    
    print(f"\n在 ω={freq} rad/s, 迎浪条件下:")
    dof_names = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
    for dof, name in enumerate(dof_names):
        unit = 'm/m' if dof < 3 else 'rad/m'
        print(f"  {name} RAO: {rao_values[dof]:.4f} {unit}")
    
    print("\n" + "=" * 70)
    print("使用说明:")
    print("=" * 70)
    print("""
在你的代码中使用:

    from load_mss_rao import load_mss_motion_rao, interpolate_rao
    
    # 加载RAO数据
    rao_data = load_mss_motion_rao('/path/to/supply.mat')
    
    # 在特定频率和浪向角处获取RAO
    rao_values = interpolate_rao(rao_data, freq=0.8, heading=np.pi)
    
    # rao_values[2] = Heave RAO (垂荡)
    # rao_values[3] = Roll RAO (横摇)  
    # rao_values[4] = Pitch RAO (纵摇)
    """)
