import numpy as np
from gym_anm.simulator import Simulator


def test_oltc_control():
    network = {
        "baseMVA": 1,
        "bus": np.array([[0, 0, 50, 1.0, 1.0], [1, 1, 50, 1.1, 0.9]]),
        "branch": np.array([[0, 1, 0.01, 0.1, 0.0, 32, 1, 0]]),
        "device": np.array(
            [
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
                [1, 0, 5, 1, 1.1, 0.9, None, None, None, None, None, None, None, None, None],
            ],
            dtype=object,
        ),
    }
    sim = Simulator(network, 1.0, 100)
    P_load = {}
    P_pot = {}
    P_set = {}
    Q_set = {}
    taps = {1: 1.05}
    sim.transition(P_load, P_pot, P_set, Q_set, taps)
    assert abs(sim.branches[(0, 1)].tap_magn - 1.05) < 1e-6
