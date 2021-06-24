import re
import os
from glob import glob
import subprocess
import numpy as np
import pandas as pd
import configparser
import matplotlib.pyplot as plt


INT_TYPES = (
    int, np.int, np.int8, np.uint8, np.int16, np.uint16,
    np.int32, np.uint32, np.int64, np.uint64
)

class XYZ:
    """
    Fast XYZ parser that can handle very large xyz file

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

        self.__kwargs.update({'skiprows': 1})
        self[0]

    def close(self):
        self.__f.close()


class TCC:
    """
    A light-weight python wrapper for Topological Cluster Classification.
    It is especially designed to handle very large xyz files.
    """
    def __init__(self, work_dir):
        self.__cwd = work_dir
        self.__raw = os.path.join(self.__cwd, 'raw_output')
        self.clusters_to_analyse = []
        self.clusters = {}

    def __len__(self):
        if self.clusters:  # not an empty cluster
            for key in self.clusters:
                return len(self.clusters[key])
        else:
            return 0

    def __write_box(self, box):
        """
        Generate a legal box.txt file in the tcc working directory
        """
        if box.ndim == 1:
            box = box[np.newaxis, :]  # (3,) --> (1, 3)
        if box.shape[1] != 3:
            raise RuntimeError("The dimension of box should be 3")
        box = np.concatenate((
            np.arange(box.shape[0])[:, np.newaxis],  # shape (n, 1)
            box  # shape (n, 3)
        ), axis=1)  # shape (n, 4)
        np.savetxt(
            fname=os.path.join(self.__cwd, 'box.txt'),
            X=box,
            header='#iter Lx Ly Lz', fmt=["%d"] + ["%.8f"] * 3
        )

    def __write_parameters(self, **kwargs):
        """
        Write inputparameters.ini and clusters_to_analyse.ini

        A symlink named `sample.xyz` should be inside self__cwd, linking
            to the xyz file to be analysed
        """
        input_parameters = {
            "Box": { "box_type": 1, "box_name": "box.txt" },
            "Run": { "xyzfilename": "sample.xyz", "frames": 1},
            "Simulation": {
                "rcutAA": 1.8, "rcutAB": 1.8, "rcutBB": 1.8, "min_cutAA": 0.0,
                "bond_type": 1, "PBCs": 1, "voronoi_parameter": 0.82,
                "num_bonds": 50, "cell_list": 1, "analyse_all_clusters": 1,
            },
            "Output": {
                "bonds": 0, "clusts": 0, "raw": 1, "do_XYZ": 0,
                "11a": 0, "13a": 0, "pop_per_frame": 1,
            }
        }

        clusters = { "Clusters": {
            "sp3a": 0, "sp3b": 0, "sp3c": 0, "sp4a": 0, "sp4b": 0, "sp4c": 0,
            "sp5a": 0, "sp5b": 0, "sp5c": 0, "6A": 0, "6Z": 0, "7K": 0,
            "7T_a": 0, "7T_s": 0, "8A": 0, "8B": 0, "8K": 0, "9A": 0, "9B": 0,
            "9K": 0, "10A": 0, "10B": 0, "10K": 0, "10W": 0, "11A": 0,
            "11B": 0, "11C": 0, "11E": 0, "11F": 0, "11W": 0, "12A": 0,
            "12B": 0, "12D": 0, "12E": 0, "12K": 0, "13A": 0, "13B": 0,
            "13K": 0, "FCC": 0, "HCP": 0, "BCC_9": 0, "BCC_15": 0,
            }
        }

        for key in kwargs:
            for section in input_parameters:
                if key in input_parameters[section].keys():
                    input_parameters[section][key] = kwargs[key]

        for key in clusters["Clusters"]:
            if key in self.clusters_to_analyse:
                clusters["Clusters"][key] = 1

        config_input = configparser.ConfigParser()
        config_input.read_dict(input_parameters)
        config_cluster = configparser.ConfigParser()
        config_cluster.read_dict(clusters)

        with open(
            os.path.join(self.__cwd, "inputparameters.ini"), 'w'
        ) as f:
            config_input.write(f)
        with open(
            os.path.join(self.__cwd, "clusters_to_analyse.ini"), 'w'
        ) as f:
            config_cluster.write(f)

    def run(self, xyz, box, frames=None, tcc_exec="tcc", silent=True, **kwargs):
        """
        Call tcc to analyse an XYZ file. The output will be write to `self.__cwd`

        Args:
            xyz (str): the path to the xyz file to be analysed. Notice that
                the working directory is `self.__cwd` if using a relative path.
            box (list): the box of the simulation / experiment. The warpper
                supports different boxes in different frames.
            frames (int): the number of frames to perform TCC analysis. This
                wrapper DOES NOT check if this value is valid.
            tcc_exec (str): the path to TCC binary executable.
            silent (bool): if True the output of TCC will be supressed
            kwargs (dict): tcc parameters. These parameters will overwrite
                the default parameters.

        Return:
            None
        """
        if self.__cwd not in os.listdir(os.getcwd()):
            os.mkdir(os.path.join(os.getcwd(), self.__cwd))
        if isinstance(frames, type(None)):
            tmp = os.getcwd()
            os.chdir(self.__cwd)
            frames = len(XYZ(xyz))
            os.chdir(tmp)
        self.__write_box(np.array(box))

        # create a soft link of the xyz file to self.__cwd
        soft_link = os.path.join(self.__cwd, "sample.xyz")
        if os.path.isfile(soft_link):
            os.remove(soft_link)
        os.symlink(src=xyz, dst=soft_link)
        self.__write_parameters(frames=frames, **kwargs)
        if silent:
            subprocess.run(
                args=tcc_exec,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.__cwd
            )
        else:
            subprocess.run(args=tcc_exec, cwd=self.__cwd)

    def parse(self):
        if not os.path.isdir(self.__raw):
            raise FileNotFoundError(
                "No raw_output folder, rerun the tcc with Output.raw = 1"
            )
        cluster_name_pattern = re.compile(r'sample\.xyz.*raw_(.+)')
        filenames = glob(
            "{folder}/sample.xyz*raw_*".format(folder=self.__raw)
        )
        filenames = [os.path.basename(fn) for fn in filenames]
        filenames.sort()
        cluster_names = [
            cluster_name_pattern.match(fn).group(1) for fn in filenames
        ]
        for cn in cluster_names:
            fn = glob(
                "{folder}/sample.xyz*raw_{cluster_name}".format(
                    folder=self.__raw, cluster_name=cn
                )
            )
            if len(fn) == 0:
                raise FileNotFoundError(
                    "Raw output file for {c} not found".format(c=cn)
                )
            if len(fn) > 1:
                raise RuntimeError(
                    "Multiple raw file found for {c}".format(c=cn)
                )
            else:
                fn = fn[0]
            self.clusters.update({
                cn: XYZ(fn, dtype=bool, true_values='C', false_values='A')
            })

    def frame(self, f):
        """
        Getting the result of particles and the clusters they are in. The
            output is a numpy array of Boolean values. One example would be

        ..code-block::

            id, FCC, 13A, 12E, 11F, 10B
            1,    0,   0,   0,   0,   1   # particle 1 is in 10B
            2,    1,   0,   0,   0,   0   # particle 2 is in FCC
            3,    0,   0,   0,   0,   0   # particle 3 is not in any cluster
            ...

        Args:
            f (int): the frame number
        """
        result_dict = {cn: xyz[f].ravel() for cn, xyz in self.clusters.items()}
        return pd.DataFrame.from_dict(data=result_dict, orient='columns')

    @property
    def population(self):
        """
        Return the mean population if each frame as a pandas table

        Return:
            pandas.DataFrame: a pandas table where each columns are the clusters
                and each rows are different frames
        """
        pattern = 'sample.xyz.*.pop_per_frame'
        fn = glob(os.path.join(self.__cwd, pattern))
        if len(fn) == 1:
            fn = fn[0]
        elif len(fn) == 0:
            raise FileNotFoundError(
                "Population output file ({p}) not found".format(p=pattern)
            )
        else:
            raise RuntimeError("Multiple population output files exist")
        df = pd.read_csv(fn, sep='\t', header=0, index_col=0)
        df.dropna(axis='columns', inplace=True)
        return df.T


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

