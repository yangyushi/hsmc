#!/usr/bin/env python
import re
import numpy as np
from itertools import product


crystal_info = {
    "cubic": {
        "unit_cell": [1, 1, 1, 90, 90, 90],
        "motif": np.array((0, 0, 0)).reshape((1, 3)),
        "lattice_points": 1,
    },
    "bcc": {
        "unit_cell": [1, 1, 1, 90, 90, 90],
        "motif": np.array(((0, 0, 0), (0.5, 0.5, 0.5))),
        "lattice_points": 2,
    },
    "fcc": {
        "unit_cell": [1, 1, 1, 90, 90, 90],
        "motif": np.array(((0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5))),
        "lattice_points": 4,
    },
    "hcp": {  # assuming close packing 
        "unit_cell": [1, 1, np.sqrt(8/3), 90, 90, 120],
        "motif": np.array(((0, 0, 0), (1.0/3.0, 1.0/3.0, 0.5))),
        "lattice_points": 2,
    },
}  # the unit is the height of the unit cell


plane_info = {
    "fcc100": {
        "unit_cell": [1, 1, 90],
        "motif": np.array(((0.0, 0.0), (0.5, 0.5)))
    },
    "fcc110": {
        "unit_cell": [1, 2**0.5, 90],
        "motif": np.array(((0.0, 0.0), (0.0, 1 / 2)))
    },
    "fcc111": {  # this is also hcp0001
        "unit_cell": [2**0.5 / 2, 6**0.5 / 2, 90],
        "motif": np.array(((0.0, 0.0), (0.5, 0.5)))
    },
}


def __get_transformation_matrix_3d(cell_vector):
    """
    this function generate a transform matrix for a set of cell_vector
    cell vector is a list, [a, b, c, <bc>, <ac>, <ab>]
    """
    a, b, c = cell_vector[:3]
    cos_bc = np.cos(cell_vector[3]/180 * np.pi)
    cos_ac = np.cos(cell_vector[4]/180 * np.pi)
    cos_ab = np.cos(cell_vector[5]/180 * np.pi)
    sin_ab = np.sin(cell_vector[5]/180 * np.pi)
    c1 = c * cos_ac
    c2 = (c * (cos_bc - cos_ab * cos_ac)) / (sin_ab)
    c3 = (c ** 2 - c1 ** 2 - c2 ** 2) ** (1/2)
    return np.array([
        [a, b*cos_ab, c1],
        [0, b*sin_ab, c2],
        [0, 0, c3]
    ])


def __index2pos_3d(indice, unit_cell):
    """
    Translate the lattice basis to Cartesian coordinates
    """
    tm = __get_transformation_matrix_3d(unit_cell)  # transform matrix
    im = np.asmatrix(indice).T  # index matrix
    pm = (tm * im).T  # position matrix
    return np.array(pm)


def __get_transformation_matrix_2d(cell_vector):
    """
    this function generate a transform matrix for a set of cell_vector
    cell vector is a list, [a, b, c, <bc>, <ac>, <ab>]
    """
    a, b = cell_vector[:2]
    gamma = cell_vector[2]/180 * np.pi
    return np.array([
        [a, b*np.cos(gamma)],
        [0, b*np.sin(gamma)]
    ])


def __index2pos_2d(indice, unit_cell):
    """
    Translate the lattice basis to Cartesian coordinates
    """
    tm = __get_transformation_matrix_2d(unit_cell)  # transform matrix
    im = np.asmatrix(indice).T  # index matrix
    pm = (tm * im).T  # position matrix
    return np.array(pm)


def get_crystal_lattice(kind, nx, ny, nz):
    """
    Get the position of common 3D crystals lattices. The lattice points
        were obtained by repeated unit cell. The unit cells were defined
        in the `crystal_info`.

    Args:
        kind (str): the type of crystals to get
        nx (int): the number of unit cells in x direction
        ny (int): the number of unit cells in y direction
        nz (int): the number of unit cells in z direction

    Return:
        numpy.ndarray: the positions of the lattice points, shape (n, 3)
    """
    indices_x = np.arange(nx)
    indices_y = np.arange(ny)
    indices_z = np.arange(nz)
    indices = np.array(list(product(indices_x, indices_y, indices_z)))
    n_indices = len(indices)
    crystal = crystal_info.get(kind)
    if crystal:
        n_motif = len(crystal['motif'])
        motif_indices = np.expand_dims(crystal['motif'], axis=0) + np.expand_dims(indices, axis=1)
        motif_indices = motif_indices.reshape((n_indices * n_motif, 3), order='F')
        pos = __index2pos_3d(motif_indices, crystal["unit_cell"])
        return pos
    else:
        raise ValueError("Invalid crystal type", kind)


def get_plane_lattice(kind, nx, ny):
    """
    Get the position of common 2D crystals lattices. The lattice points
        were obtained by repeated unit cell. The unit cells were defined
        in the `plane_info`.

    Args:
        kind (str): the type of crystals to get
        nx (int): the number of unit cells in x direction
        ny (int): the number of unit cells in y direction

    Return:
        numpy.ndarray: the positions of the lattice points, shape (n, 3)
    """
    indices_x = np.arange(nx)
    indices_y = np.arange(ny)
    indices = np.array(list(product(indices_x, indices_y)))
    n_indices = len(indices)
    plane = plane_info.get(kind)
    if plane:
        n_motif = len(plane['motif'])
        motif_indices = np.expand_dims(plane['motif'], axis=0) + np.expand_dims(indices, axis=1)
        motif_indices = motif_indices.reshape((n_indices * n_motif, 2), order='F')
        pos = __index2pos_2d(motif_indices, plane["unit_cell"])
        return pos
    else:
        raise ValueError("Invalid crystal type", kind)


def parse_plane_kind(kind):
    """
    Get the crystal name from the kind of the plane

    Args:
        kind (str): the type of plane to get, e.g. fcc111

    Return:
        str: the type of corresponding crystal, e.g. fcc
    """
    crystal_pattern = re.match(r'([a-z]+)\d*', kind.lower())
    if crystal_pattern:
        crystal_kind = crystal_pattern.group(1)
        is_valid = (crystal_kind in crystal_info) and (kind in plane_info)
        if is_valid:
            return crystal_kind
        else:
            raise ValueError("Invalid plane type: " + kind)
    else:
        raise ValueError("Can't find crystal from token: " + kind)


def get_lattice_constant(kind, vf, sigma=1.0):
    """
    Get the lattice constant of a crystal at given volume fraction

    Args:
        kind (str): the type of crystal, e.g. fcc
        vf (float): the volume fraction, e.g. 0.545
        sigma (float): the diameter of the particles.

    Result:
        float: the lattice constant
    """
    n = crystal_info[kind]['lattice_points']
    a = np.power(n * np.pi * sigma**3 / 6.0 / vf, 1.0 / 3.0)
    return a


def get_plane(kind, nx, ny, vf, sigma=1, report=True):
    """
    Get the position of common crystal planes. The particles were obtained
        by repeated unit cell. The unit cells were defined in the `plane_info`,
        whose size is determined by the volume fraction.

    Args:
        kind (str): the type of plane to get, e.g. fcc111
        nx (int): the number of unit cells in x direction.
        ny (int): the number of unit cells in y direction.
        vf (float): the corresponding volume fraction of the 3D crystal.
        sigma (float): the diameter of the particles

    Return:
        tuple: (positions, box)
            - the positions of the particles, shape (n, 3)
            - box: the box that contains the particles
    """
    crystal_kind = parse_plane_kind(kind)
    a = get_lattice_constant(crystal_kind, vf=vf, sigma=sigma)
    lattice = get_plane_lattice(kind, nx, ny)
    box = np.array((nx, ny)) * a * np.array(plane_info[kind]['unit_cell'][:2])
    if report:
        print(f"Creating {kind} plane with size of {box[0]:.4f} x {box[1]:.4f}, N = {len(lattice)}")
    return lattice * a, box


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for name, crystal in crystal_info.items():
        pos = get_crystal_lattice(name, 3, 3, 2)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(name)
        ax.scatter(*pos.T)
        plt.show()

    for name, crystal in plane_info.items():
        pos = get_plane_lattice(name, 3, 3)
        plt.figure(figsize=(4, 4))
        plt.title(name)
        plt.scatter(*pos.T, s=100)
        plt.tight_layout()
        plt.tight_layout()
        plt.show()
