# Hard Sphere Monte Carlo Simulation

This is my personal project to study Monte-Carlo simulations of monodispersed hard spheres in 3D.

It is not very fast. If you need large scale simulation, maybe you want to use [hoomd-blue](https://github.com/glotzerlab/hoomd-blue).


## Install

You need to have [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) and [pybind11](https://pybind11.readthedocs.io/en/stable/) to build the code. The project can be built with CMake, normally you will do,

```sh
mkdir build
cd build
cmake ..
make
make install
```

As a result, the python module (with name `chard_sphere.cpython-xx-xxx.so`) will be created in folder `lib`, as well as the static C++ library (`libhard_sphere.a`).

## Python Frontend

The user is expceted to interact with the library via the Python interface. Typically, one shuold

1. copy the file `chard_sphere.cpython-xx-xxx.so` to the working directory; or
2. add the folder `lib` to `$PYTHONPATH`.

After that, you should be able to execute the command `from hsmc import chard_sphere` in Python.


## Use the Code in Python

The following snippest present the way to start a simulation

```py
from hsmc import chard_sphere
from hsmc.analysis import dump_xyz, TCCOTF
import numpy as np


n_particle = 1000
box = [30, 50, 60]  # box size in X, Y, and Z
is_pbc = [True, True, False]  # no PBC in the z-direction
is_hard = [False, False, True]  # hard walls along z-direction
r_skin = 5.0

# create the system
system = chard_sphere.HSMC(
    n=n_particle, box=box,
    is_pbc=is_pbc, is_hard=is_hard, r_skin=r_skin
)

# randomly fill non-overlapping hard spheres
system.fill_hs()

# reduce the box size slowly to reach desired volume fraction
system.crush(0.50, 0.02)

# print the overview of the system
print(system)


# perform 1000 Monte-Carlo sweeps, collect the configuration every 100 frames

configurations = np.empty((10, n_particle, 3))
for i in range(10):
    for _ in range(100):
        system.sweep()

    # retrieve the positions of the particles
    positions = system.get_positions()
    configurations[i] = positions.T

    # save the positions into an xyz file
    dump_xyz(
        'hard_sphere.xyz', positions.T,
        comment="box:[" + ",".join([  # write box as the comment
            f"{L:.8f} (PBC? {p}; Hard? {h})"
            for L, p, h in zip(system.get_box(), is_pbc, is_hard)
        ]) + "]"
    )
```
