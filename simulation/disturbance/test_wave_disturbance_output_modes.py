import os
import sys

import numpy as np

# Allow running this script from any CWD by injecting the project root
# (the directory that contains the `simulation/` package) into sys.path.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulation.disturbance.wave_disturbance import WaveDisturbance
from simulation.common.platform_dynamics import ParallelPlatform3DOF


def test_output_modes_shapes():
    t = np.linspace(0.0, 2.0, 2001)

    wave = WaveDisturbance(
        Hs=2.0,
        T1=8.0,
        vessel_file='supply.mat',
        n_components=20,
        random_seed=0,
        enable_burst_step=False,
    )

    ship = wave.generate(t, output='ship_state')
    assert ship['q_s'].shape == (len(t), 6)
    assert ship['qd_s'].shape == (len(t), 6)
    assert ship['qdd_s'].shape == (len(t), 6)

    platform = ParallelPlatform3DOF()
    # simple nominal platform total state (all zeros except z)
    q_u = np.zeros((len(t), 3))
    q_u[:, 0] = 1.058
    qd_u = np.zeros((len(t), 3))

    out = wave.generate(t, output='tau_dist', q_u=q_u, qd_u=qd_u, platform=platform)
    assert out['tau_dist'].shape == (len(t), 3)


if __name__ == '__main__':
    test_output_modes_shapes()
    print('OK')
