"""
Common modules for thesis simulation
共享模块，被各章节仿真代码调用
"""

from .platform_dynamics import ParallelPlatform3DOF
from .wave_disturbance import WaveDisturbance

__all__ = ['ParallelPlatform3DOF', 'WaveDisturbance']
