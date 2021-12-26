import sys
sys.path.append("../lib")
import hsmc
import numpy as np


def test_hcp100():
    plane = hsmc.crystal.get_plane_lattice('hcp110', 5, 5)
    xtal = hsmc.crystal.get_crystal_lattice('hcp', 12, 5, 5)
    x, y, z = xtal.T


    rot = np.array((
        (0.5, -3**0.5/2),
        (3**0.5/2, 0.5),
    ))
    x, y = rot @ np.array((x, y))
    xtal = np.array((x, y, z)).T

    s = 0
    mask = xtal.T[0] < s+0.1
    mask *= xtal.T[0] > s-1e-8
    plane_cut = xtal[mask]
    plane_cut -= plane_cut.min(axis=0)

    assert np.allclose(plane, plane_cut[:, 1:]), "hcp 110 plane is not correct"


if __name__ == "__main__":
    test_hcp100()
