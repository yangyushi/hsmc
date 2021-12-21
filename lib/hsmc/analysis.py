import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INT_TYPES = (
    int, np.int, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64
)


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


class FrameIter:
    """
    Iter frame-wise data in a text file, organised as

    Frame 1
    Frame 2
    ...

    For each frame, the content is

    Header   # one-line stating start of frame
    Comment  # many-lines arbitrary texts to be skipped
    Data     # many-lines data to be loaded as a numpy array

    A typical application for this is to parse the XYZ file.


    Attributes:
        numbers (list): the number of particles in each frame
        __f (io.FileIO): a FileIO instance obtained by `open()`
        __frame_cursors (list): the stream position of the start of each frame,
            the cursor is located at the start of the 2nd line of the frame.
            (the comment line)
        __kwargs (dict): the arguments to load a single frame using function\
            `numpy.loadtxt` or `pandas.read_csv`.
        __engine (str): choose the engine to load the result to numpy array\
            [pandas]: the data were obtained with `pandas.read_csv`;\
            [numpy]: the data were obtained with `numpy.loadtxt`.\
            (in 2021, the pandas engine were ~10x faster)
        __func (callable): the function to be called to obtain results
    """
    def __init__(self, filename, header_pattern, n_comment, engine='pandas', **kwargs):
        self.numbers = []
        self.__frame = 0
        self.__frame_cursors = []
        self.__header_pattern = header_pattern
        self.__n_comment = n_comment
        self.__filename = filename
        self.__kwargs = {}
        self.__f = open(filename, 'r')

        if engine.lower() in ['pandas', 'pd', 'p']:
            self.__engine = 'pandas'
        elif engine.lower() in ['numpy', 'np', 'n']:
            self.__engine = 'numpy'
        self.__parse()
        self.__set_load_parameters(**kwargs)
        self.ndim = self.__detect_dimension()

    def __next__(self):
        if self.__frame < len(self):
            self.__frame += 1
            return self[self.__frame - 1]
        else:
            self.__frame = 0
            raise StopIteration

    def __getitem__(self, i):
        """
        Args:
            i (int): the frame number

        Return:
            np.ndarray: the information of all particles in a frame, shape (n, dim)
        """
        if type(i) in INT_TYPES:
            if self.numbers[i] == 0:
                return np.empty((0, self.ndim))
            self.__f.seek(self.__frame_cursors[i])
            if self.__engine == 'pandas':
                result = self.__func(
                    self.__f, nrows=self.numbers[i], **self.__kwargs
                ).values
            elif self.__engine == 'numpy':
                result = self.__func(
                    self.__f, max_rows=self.numbers[i], **self.__kwargs
                )
            else:
                raise ValueError("Unknown engine name, select from [numpy] or [pandas]")
            if result.ndim == 1:  # for frames with just 1 particle
                return result[np.newaxis, :]
            else:
                return result
        elif type(i) == slice:
            result = []
            start = i.start if i.start else 0
            stop = i.stop if i.stop else len(self)
            step = i.step if i.step else 1
            for frame in range(start, stop, step):
                result.append(self[frame])
            return result

    def __len__(self):
        """
        return the total number of frames
        """
        return len(self.numbers)

    def __iter__(self): return self

    def __set_load_parameters(self, **kwargs):
        """
        this method is used to handle the difference between the
            two engines.
        """
        if 'delimiter' not in kwargs:  # use space as default delimiter
            kwargs.update({'delimiter': ' '})
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
        self.__kwargs['skiprows'] = self.__n_comment  # skip the comment

    def __parse(self):
        self.__frame_cursors = []
        self.numbers = []
        self.__f.seek(0)
        line = self.__f.readline()
        numbers = 0
        while line:
            is_head = re.match(self.__header_pattern, line)
            if is_head:
                self.numbers.append(numbers - self.__n_comment)
                cursor = self.__f.tell()
                self.__frame_cursors.append(cursor)
                numbers = 0
            else:
                numbers += 1
            line = self.__f.readline()
        self.numbers.pop(0)
        self.numbers.append(numbers)  # add the last frame

    def close(self):
        self.__f.close()

    def __del__(self):
        self.__f.close()

    def __detect_dimension(self):
        for i, num in enumerate(self.numbers):
            if num > 0:
                self.__f.seek(self.__frame_cursors[i])
                if self.__engine == 'pandas':
                    result = self.__func(
                        self.__f, nrows=self.numbers[i], **self.__kwargs
                    ).values
                elif self.__engine == 'numpy':
                    result = self.__func(
                        self.__f, max_rows=self.numbers[i], **self.__kwargs
                    )
                else:
                    raise ValueError("Unknown engine name, select from [numpy] or [pandas]")
                return result.shape[1]
        return 0

    def to_json(self, filename=None):
        """
        Save the essential data in the calss.

        Args:
            filename (bool or str): if the filename is None, the data will\
                be returned as a dictionary. If filename is str, the data\
                will be write to the harddrive.

        Return:
            None or dict: the essential data to reconstruct the object.
        """
        data = {
            'numbers': self.numbers,
            'filename': self.__filename,
            'frame': self.__frame,
            'frame_cursors': self.__frame_cursors,
            'header_pattern': self.__header_pattern,
            'n_comment': self.__n_comment,
            'engine': self.__engine,
            'ndim': self.ndim,
            'kwargs': self.__kwargs,  # TODO: ensure elements are serialisable
        }
        if isinstance(filename, type(None)):
            return data
        elif isinstance(filename, str):
            with open(filename, 'w') as f:
                json.dump(data, f)

    @classmethod
    def from_json(cls, data):
        """
        Create a frame iterable without parsing the file.\
            Instead load the metadata from a dict, or a\
            json file on the disk.

        Args:
            data (dict or str): a dictionary containing all elements\
            (see `FrameIter.to_json`), or a string to the json file\
            containing the dict.

        Example:
            >>> obj = FrameIter('sample.xyz')

            >>> cache = obj.to_json()  # save data in memory
            >>> new_obj = FrameIter.from_json(cache)

            >>> obj.to_json("cache.json")  # save data in disk
            >>> new_obj = FrameIter.from_json("cache.json")
        """
        if isinstance(data, str):
            with open(data, 'r') as f:
                data = json.load(f)
        elif isinstance(data, dict):
            pass
        else:
            raise TypeError(
                "Invalid datatype"
            )
        self = cls.__new__(cls)  # bypass __init__
        self.filename = data['filename']
        self.numbers = data['numbers']
        self.__frame = data['frame']
        self.__frame_cursors = data['frame_cursors']
        self.__header_pattern = data['header_pattern']
        self.__n_comment = data['n_comment']
        self.__engine = data['engine']
        if self.__engine == 'numpy':
            self.__func = np.loadtxt
        elif self.__engine == 'pandas':
            self.__func = pd.read_csv
        else:
            raise ValueError(
                "Unknown engine name, select from [numpy] or [pandas]"
            )
        self.ndim = data['ndim']
        self.__kwargs = data['kwargs']
        self.__f = open(self.filename)
        return self


class XYZ(FrameIter):
    """
    Fast XYZ parser that can handle very large xyz file

    """
    def __init__(self, filename, engine='pandas', align_opt=False, **kwargs):
        """
        Args:
            filename (str): the path to the xyz file to be loaded.
            engine (str): choose between pandas or numpy, pandas is faster
            align_opt (bool): Significantly (!) optimise the parsing speed\
                if the data in the xyz file is *right-aligned*, meaning\
                all coordinates have the same column width. If the \
                optimisation was mistakenly used for not aligned data,\
                an runtime error will be raised.
        """
        if align_opt: self._FrameIter__parse = self.__fast_parse
        super().__init__(
            filename,
            header_pattern=r'(\d+)\n',
            n_comment=1,
            engine=engine,
            **kwargs,
        )

    def __detect_line_offset(self):
        """
        Find the byte offset one one line in the data.

        Exampe:
            >>> lines_to_jump = 1000
            >>> offset = self.__detect_line_offset()
            >>> new_location = self.__f.tell() + offset * 1000
            >>> self.__f.seek(new_location)
        """

        f = self._FrameIter__f
        hp = self._FrameIter__header_pattern
        nc = self._FrameIter__n_comment
        f.seek(0)

        line = f.readline()
        while line:
            is_head = re.match(hp, line)
            if is_head:
                for _ in range(nc):
                    f.readline()
                cursor_before_line = f.tell()
                line = f.readline()
                if not re.match(hp, line):
                    return f.tell() - cursor_before_line
                else:
                    line = f.readline()
        raise RuntimeError("Can't detect the line offset")

    def __fast_parse(self):
        lo = self.__detect_line_offset()
        self._FrameIter__frame_cursors = []
        fcs = self._FrameIter__frame_cursors
        f = self._FrameIter__f
        hp = self._FrameIter__header_pattern
        nc = self._FrameIter__n_comment

        self.numbers = []
        f.seek(0)
        line = f.readline()

        while line:
            is_head = re.match(hp, line)
            if is_head:
                cursor = f.tell()
                fcs.append(cursor)
                n_particle = int(re.match('(\d+)\n', line).group(0))
                self.numbers.append(n_particle)
                for _ in range(nc):
                    f.readline()
                f.seek(f.tell() + n_particle * lo)
                line = f.readline()
            else:
                raise RuntimeError(
                    "Failed to parse the xyz file with align optimisation"
                )


def __isf_3d(x1, x2, pbc_box=[None, None, None], q=np.pi*2):
    """
    Calculate the self intermediate scattering function between two configurations.

    Args:
        x1 (numpy.ndarray): the particle locations, shape (n, 3)
        x2 (numpy.ndarray): the particle locations, shape (n, 3)
        pbc_box (iterable): the side length of a periodic boundary, if there\
            is no PBC in z-direction, then pbc_box should be [Lx, Ly, None].
        q (float): the wavenumber.

    Return
        flaot: the value of the self intermediate scattering function
    """
    shift = x2 - x1
    for d in range(3):
        if not isinstance(pbc_box[d], type(None)):
            s1d = shift[:, d]
            mask_1 = s1d > pbc_box[d] / 2.0
            mask_2 = s1d < - pbc_box[d] / 2.0
            shift[:, d][mask_1] -= pbc_box[d]
            shift[:, d][mask_2] += pbc_box[d]
    shift -= shift.mean(0)[np.newaxis, :]
    dist = np.linalg.norm(shift, axis=1)
    f = np.sinc((q / np.pi) * dist)
    return np.mean(f)


def get_isf_3d(trajectory, pbc_box, q=2*np.pi, length=None, sample_num=None):
    """
    Calculate the average isf from trajectory

    Args:
        trajectory (iterable): a collection of positions arranged according\
            to the time.
        pbc_box (iterable): the side length of a periodic boundary, if there\
            is no PBC in z-direction, then pbc_box should be [Lx, Ly, None].
        q (float): the wavenumber.
        legnth (int): the largest lag time of the isf.
        sample_num (int): the maximum number of points sampled per tau value.

    Return:
        numpy.ndarray: the isf as a function of lag time.
    """
    length_full = len(trajectory)

    if isinstance(length, type(None)):
        length = length_full

    if isinstance(sample_num, type(None)):
        sample_num = length

    isf = np.zeros(length)
    count = np.zeros(length)
    for i in range(length_full):
        for j in range(i+1, length_full):
            tau = j - i
            if (count[tau] == sample_num) or (tau > length):
                continue
            isf[tau] += __isf_3d(trajectory[i], trajectory[j], pbc_box, q)
            count[tau] += 1
    count[0] = 1
    isf[0] = 1
    return isf / count
