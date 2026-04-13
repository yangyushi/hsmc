# Hard Sphere Monte Carlo Simulation

This is my personal project to study Monte-Carlo simulations of monodispersed hard spheres in 3D.

It is not very fast. If you need large scale simulation, maybe you want to use [hoomd-blue](https://github.com/glotzerlab/hoomd-blue).


## Install

The intended installation flow is now:

```sh
pip install .
```

This builds and installs:

- the Python package `hsmc`
- the compiled extension module `hsmc.chard_sphere`
- the bundled assets under `workflow`
- the helper command `hsmc-create-slit`
- the helper command `hsmc-create-slit-v2`

After installation, the package should be importable directly:

```py
import hsmc
from hsmc import chard_sphere
```

The build still requires [Eigen](https://eigen.tuxfamily.org/index.php?title=Main_Page) to be available for CMake to find during installation.


## Bundled Assets

The slit simulation templates are shipped with the installed package. After
installation, you can create a local copy of the original workflow with:

```sh
hsmc-create-slit
```

This creates a `simulate_slit/` directory in the current working directory.

The refactored single-entrypoint workflow is available separately with:

```sh
hsmc-create-slit-v2
```

This creates a `simulate_slit_v2/` directory in the current working directory.


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
