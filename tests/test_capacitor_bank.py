import numpy as np
from gym_anm.simulator import Simulator


def test_capacitor_bank_control():
    network = {
        "baseMVA": 1,
        "bus": np.array([[0, 0, 50, 1.0, 1.0], [1, 1, 50, 1.1, 0.9]]),
        "branch": np.array([[0, 1, 0.01, 0.1, 0.0, 32, 1, 0]]),
        "device": np.array(
            [
                [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
                [1, 1, 4, None, 0, 0, 1, 0, None, None, None, None, None, None, None],
            ]
        ),
    }
    sim = Simulator(network, 1.0, 100)
    P_load = {}
    P_pot = {}
    P_set = {}
    Q_set = {1: 0.5}
    sim.transition(P_load, P_pot, P_set, Q_set)
    assert abs(sim.devices[1].q - 0.5) < 1e-6
