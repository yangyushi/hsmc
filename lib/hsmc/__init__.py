import numpy as np
from . import analysis, crystal
from . import chard_sphere
from .analysis import XYZ, TCC


def dump_xyz(filename, positions, comment=''):
    """
    Dump positions into an xyz file

    Args:
        filename (str): the name of the xyz file, it can be an existing file
        positions (numpy.ndarray): the positions of particles, shape (n, dim)

    Return:
        None
    """
    n, dim = positions.shape
    with open(filename, 'a') as f:
        np.savetxt(
            f, positions, delimiter=' ',
            header='%s\nframe %s' % (n, comment),
            comments='',
            fmt=['A %.8e'] + ['%.8e' for i in range(dim - 1)]
        )
