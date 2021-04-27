# Hard Sphere Monte Carlo Simulation


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


n_particle = 500
box = [10, 10, 10]  # box size in X, Y, and Z
is_pbc = [True, True, False]  # no PBC in the z-direction

# create the system
system = chard_sphere.HSMC(n_particle, box, is_pbc)

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
```

## Other utilities in Python Module

The `hsmc` module in the lib folder contains other handy tools. For instance a very fast `XYZ` file parser, and some crystallography related functions.
