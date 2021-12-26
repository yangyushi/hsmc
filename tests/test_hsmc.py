import sys
sys.path.append('../lib')
from hsmc import chard_sphere


def test_hsmc():
    n_particle = 500
    box = [10, 10, 10]  # box size in X, Y, and Z
    is_pbc = [True, True, False]  # no PBC in the z-direction
    is_hard = [False, False, True]  # hard walls in z-direction

    # create the system
    system = chard_sphere.HSMC(n_particle, box, is_pbc, is_hard)

    # randomly fill non-overlapping hard spheres
    system.fill_hs()

    # reduce the box size slowly to reach desired volume fraction
    system.crush(0.4, 0.02)

    # print the overview of the system
    print(system)

    # perform 100 Monte-Carlo sweeps
    for _ in range(100):
        system.sweep()

    # retrieve the positions of the particles
    positions = system.get_positions()

    assert int(positions.shape[1]) == n_particle
    assert int(positions.shape[0]) == 3
    assert not system.report_overlap(), "Overlap detected"


if __name__ == "__main__":
    test_hsmc()
