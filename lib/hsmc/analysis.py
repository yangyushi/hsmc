import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INT_TYPES = (
    int, np.int, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64
)

class XYZ:
    """
    Very fast XYZ parser that can handle very large xyz file

    Attributes:
        particle_numbers (list): the number of particles in each frame
        __f (io.FileIO): a FileIO instance obtained by `open()`
        __frame_cursors (list): the stream position of the start of each frame,
            the cursor is located at the start of the 2nd line of the frame.
            (the comment line)
        __kwargs (dict): the arguments to load a single frame using function
            `numpy.loadtxt`.
        __engine (str): choose the engine to load the result to numpy array
            [pandas]: the data were obtained with `pandas.read_csv`;
            [numpy]: the data were obtained with `numpy.loadtxt`.
            (in 2021, the pandas engine were ~10x faster)
        __func (callable): the function to be called to obtain results
    """
    def __init__(self, filename, engine='pandas', **kwargs):
        self.particle_numbers = []
        self.__f = open(filename, 'r')
        self.__frame = 0
        self.__frame_cursors = []
        if engine.lower() in ['pandas', 'pd', 'p']:
            self.__engine = 'pandas'
        elif engine.lower() in ['numpy', 'np', 'n']:
            self.__engine = 'numpy'
        self.__parse()
        self.set_load_parameters(**kwargs)

    def __parse(self):
        self.__frame_cursors = []
        self.particle_numbers = []
        self.__f.seek(0)
        line = self.__f.readline()
        while line:
            is_head = re.match(r'(\d+)\n', line)
            if is_head:
                cursor = self.__f.tell()
                particle_num = int(is_head.group(1))
                self.__frame_cursors.append(cursor)
                self.particle_numbers.append(particle_num)
                for _ in range(particle_num):
                    self.__f.readline()
            line = self.__f.readline()

    def __len__(self):
        """
        return the total number of frames
        """
        return len(self.particle_numbers)

    def __iter__(self): return self

    def __next__(self):
        if self.__frame < len(self):
            self.__frame += 1
            return self[self.__frame - 1]
        else:
            self.__frame = 0
            raise StopIteration

    def __getitem__(self, i):
        if type(i) in INT_TYPES:
            self.__f.seek(self.__frame_cursors[i])
            if self.__engine == 'pandas':
                result = self.__func(
                    self.__f, nrows=self.particle_numbers[i], **self.__kwargs
                ).values
            elif self.__engine == 'numpy':
                result = self.__func(
                    self.__f, max_rows=self.particle_numbers[i], **self.__kwargs
                )
            else:
                raise ValueError("Unknown engine name, select from [numpy] or [pandas]")
            return result
        elif type(i) == slice:
            result = []
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self)
            step = i.step if i.step else 1
            for frame in range(start, stop, step):
                result.append(self[frame])
            return result

    def set_load_parameters(self, **kwargs):
        if self.__engine == 'numpy':
            self.__func = np.loadtxt
            self.__kwargs = {
                key : val for key, val in kwargs.items()
                if key not in ['skiprows', 'max_rows']
            }
        elif self.__engine == 'pandas':
            self.__func = pd.read_csv
            self.__kwargs = {
                key : val for key, val in kwargs.items()
                if key not in ['skiprows', 'nrows']
            }
            self.__kwargs.update({'index_col': False, 'header': None})
        else:
            raise ValueError(
                "Unknown engine name, select from [numpy] or [pandas]"
            )

        self.__kwargs.update({'skiprows': 1})
        self[0]

    def close(self):
        self.__f.close()


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
    z = []
    for i, frame in enumerate(frames):
        z.append(frame[:, -1])
    z = np.concatenate(z)
    z_mid = z.mean()
    z_range = np.linspace(1, z_mid / 2, npoints)
    n_frames = int(np.ceil(len(frames) / jump))
    bulk_vf_ensemble = np.empty((n_frames, npoints))
    for f, frame in enumerate(frames[::jump]):
        bulk_vf = np.empty(npoints)
        for i, dz in enumerate(z_range):
            bulk_vf[i] = get_slice_vf(
                frame, z_mid - dz, z_mid + dz, box
            )
        bulk_vf_ensemble[f] = bulk_vf

    weight = np.exp(-z_range)
    weight = weight / weight.sum() * bulk_vf_ensemble.shape[1]
    vf = (bulk_vf_ensemble.mean(axis=0) * weight).mean()
    if plot:
        plt.errorbar(
            x = z_range * 2, y=np.mean(bulk_vf_ensemble, axis=0),
            yerr=np.std(bulk_vf_ensemble, axis=0) / np.sqrt(bulk_vf_ensemble.shape[0]),
            color='k', marker='o'
        )

        plt.plot((z_range[0], z_range[-1] * 2), [vf] * 2, color='k', ls='--', label=f"Bulk {vf:.4f}")
        plt.gcf().set_size_inches(8, 4)
        plt.xlabel("Central Region Thickness")
        plt.ylabel("Bulk Volumn Fraction")
        plt.legend()
        plt.tight_layout()
        if save:
            plt.savefig(save)
        else:
            plt.show()
    return vf
