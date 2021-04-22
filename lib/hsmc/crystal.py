from itertools import product
import numpy as np

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
    "hcp": {
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


def get_transformation_matrix_3d(cell_vector):
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


def index2pos_3d(indice, unit_cell):
    """
    Translate the lattice basis to Cartesian coordinates
    """
    tm = get_transformation_matrix_3d(unit_cell)  # transform matrix
    im = np.asmatrix(indice).T  # index matrix
    pm = (tm * im).T  # position matrix
    return np.array(pm)


def get_transformation_matrix_2d(cell_vector):
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


def index2pos_2d(indice, unit_cell):
    """
    Translate the lattice basis to Cartesian coordinates
    """
    tm = get_transformation_matrix_2d(unit_cell)  # transform matrix
    im = np.asmatrix(indice).T  # index matrix
    pm = (tm * im).T  # position matrix
    return np.array(pm)


def get_crystal(kind, nx, ny, nz):
    """
    Get the position of common 3D crystals
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
        pos = index2pos_3d(motif_indices, crystal["unit_cell"])
        return pos
    else:
        raise ValueError("Invalid crystal type", kind)


def get_plane(kind, nx, ny):
    """
    Get the position of common 2D crystal planes
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
        pos = index2pos_2d(motif_indices, plane["unit_cell"])
        return pos
    else:
        raise ValueError("Invalid crystal type", kind)



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    for name, crystal in crystal_info.items():
        pos = get_crystal(name, 3, 3, 2)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_title(name)
        ax.scatter(*pos.T)
        plt.show()

    for name, crystal in plane_info.items():
        pos = get_plane(name, 3, 3)
        plt.figure(figsize=(4, 4))
        plt.title(name)
        plt.scatter(*pos.T, s=100)
        plt.tight_layout()
        #plt.xlim(0, 4)
        #plt.ylim(0, 4)
        plt.tight_layout()
        plt.show()
