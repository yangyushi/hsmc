import numpy as np
import matplotlib.pyplot as plt


def get_slice_vf(positions, s_min, s_max, box, axis=2, sigma=1.0):
    """
    Calculate the volumn fraction in range (s_min, s_max).

    See the wiki (https://en.wikipedia.org/wiki/Spherical_cap)
    for the equation

    ..code-block::

                │   case 1  │
                │   ,───.   │
         case 2 │  (  +  )  │ case 3
             ,──┼.  `───'  ,┼──.
            (  +│ )       ( │+  )
             `──┼'         `┼──'
               ,┼──.     ,──┼.
       case 4 ( │+  )   (  +│ ) case 5
               `┼──'     `──┼'
                │           │
              s_min       s_max


    Args:
        positions (np.ndarray): a collection of 3D data shape (n, 3)
        s_min (float): the minimum value of the slice
        s_max (float): the maximum value of the slice
        box (list or tuple): the boundary, (box_x, box_y, box_z)
        axis (int): the axis normal to the slicing planes
        sigma (float): the diameter of the particles

    Return:
        float: the volumn fraction inside the selection
    """
    r = sigma / 2.0
    z = positions[:, axis]
    mask_full = np.logical_and( # entire sphere inside region, case 1
        z < s_max - r, z > s_min + r
    )
    mask_minor = np.logical_or( # minority of the sphere inside region
        np.logical_and(z > s_min - r, z < s_min), # case 2
        np.logical_and(z < s_max + r, z > s_max)  # case 3
    )
    mask_major = np.logical_or( # majority of sphere inside region
        np.logical_and(z > s_min, z < s_min + r), # case 4
        np.logical_and(z < s_max, z > s_max - r)  # case 5
    )
    h = np.min(  # distance between z and closest boundary
        (np.abs(s_max - z),  np.abs(z - s_min)),
        axis=0
    )
    volumn_box = box[0] * box[1] * (s_max - s_min)
    volumn_sphere_full = np.pi / 6.0 * np.sum(mask_full) * sigma**3
    volumn_sphere_major = np.sum(
        np.pi * (h + r)**2 / 3.0 * (3 * r - (h + r)) * mask_major
    )
    volumn_sphere_minor = np.sum(
        np.pi * h**2 / 3.0 * (3 * r - h) * mask_minor
    )
    volumn_sphere = volumn_sphere_full + volumn_sphere_major + volumn_sphere_minor
    return volumn_sphere / volumn_box


def get_bulk_vf(frames, box, jump, npoints=50, plot=True, save="state-point.pdf"):
    """
    Find the bulk volume fraction in the central region of a slit

    Args:
        frames (iterable): a collection of different particles positions, each
            element is a (n, 3) numpy array
        box (iterable): three numbers indicating the x, y, z size of the simulation box
        jump (int): every [jump]th frame will be used in the calculation.
        npoints (int): the number of slice thickness to be calculated.
        plot (bool): if true, generate a plot.
        save (str): if it is not empty, the plot will be saved using [save] as filename

    Return:
        float: the volumn fraction corresponding to the central region
    """
    z_mid = 0
    for i, frame in enumerate(frames):
        z_mid += frame[:, -1].mean()
    z_mid /= len(frames)
    z_range = np.linspace(1, z_mid / 2, npoints)
    n_frames = int(np.ceil(len(frames) / jump))
    bulk_vf = np.zeros(npoints)
    count = 0
    for f, frame in enumerate(frames):
        if f % jump == 0:
            for i, dz in enumerate(z_range):
                bulk_vf[i] += get_slice_vf(
                    frame, z_mid - dz, z_mid + dz, box
                )
            count += 1

    bulk_vf /= count
    weight = np.exp(-z_range)
    weight = weight / weight.sum() * npoints
    vf = (bulk_vf * weight).mean()
    if plot:
        plt.scatter(
            x = z_range * 2, y=bulk_vf,
            color='w', marker='o', ec='k'
        )

        plt.plot(
            (z_range[0], z_range[-1] * 2), [vf] * 2,
            color='k', ls='--', label="Bulk {vf:.4f}".format(vf=vf)
        )
        plt.gcf().set_size_inches(8, 4)
        plt.xlabel("Central Region Thickness")
        plt.ylabel("Bulk Volumn Fraction")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save)
        else:
            plt.show()
        plt.close()
    return vf


def dump_xyz(filename, positions, comment=''):
    """
    Dump positions into an xyz file

    Args:
        filename (str): the name of the xyz file, it can be an existing file
        positions (numpy.ndarray): the positions of particles, shape (n, dim)
        comment (str): the content in the comment line

    Return:
        None
    """
    n, dim = positions.shape
    with open(filename, 'a') as f:
        np.savetxt(
            f, positions, delimiter=' ',
            header='%s\n%s' % (n, comment),
            comments='',
            fmt=['A %.8e'] + ['%.8e' for i in range(dim - 1)]
        )
