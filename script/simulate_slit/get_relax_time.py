#!/usr/bin/env python3
import os
import json
import numpy as np
import configparser
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import hsmc



conf = configparser.ConfigParser()
conf.read('configure.ini')

length = int(conf['ISF']['length'])
jump = int(conf['ISF']['jump'])
show_isf = bool(conf['ISF']['show_isf'])
plot_isf = bool(conf['ISF']['plot_isf'])

valid_planes = ('fcc100', 'fcc111', 'fcc110')

# load parameters
N = int(conf['ISF']['n'])
sigma = 1
vf_init = float(conf['System']['vf_init'])
vf_final = float(conf['System']['vf_final'])
r_skin = float(conf['System']['r_skin'])
vf_crystal = float(conf['Boundary']['vf_crystal'])
z_final = float(conf['Boundary']['z'])
kind = conf['Boundary']['kind']


v_sph = np.pi * N * sigma ** 3 / 6
l_xy = np.sqrt(v_sph / vf_final / z_final)  # approx value

# setup the boundary
if kind == 'hardwall':
    z_init = v_sph / vf_init / l_xy ** 2
    box = np.array((l_xy, l_xy, z_init))
    configuration = np.random.uniform(0, 1, (N, 3)) * box
    indices_to_move = np.arange(N)

elif kind in valid_planes:
    crystal_kind = hsmc.crystal.parse_plane_kind(kind)
    a = hsmc.crystal.get_lattice_constant(
        crystal_kind, vf=vf_crystal, sigma=sigma
    )
    uc = np.array(hsmc.crystal.plane_info[kind]['unit_cell'][:2])
    nx = int(np.floor(l_xy / (uc[0] * a)))
    ny = int(np.ceil(l_xy / (uc[1] * a)))

    plane, box_xy = hsmc.crystal.get_plane(
        kind, nx, ny, vf=vf_crystal, sigma=sigma
    )

    # (x, y) --> (x, y, 0)
    plane = np.concatenate((plane, np.zeros((len(plane), 1))), axis=1)

    # zoom z to desired volumn fraction
    v_sph = np.pi * (N + len(plane) * 2) * sigma ** 3 / 6
    a_xy = box_xy[0] * box_xy[1]  # area of box in xy plane
    box_z = v_sph / a_xy / vf_init

    # concatenate top and bottom planes
    walls = np.concatenate((plane, plane + np.array((0, 0, box_z))), axis=0)

    box = np.array((box_xy[0], box_xy[1], box_z))

    gas = np.random.uniform(0, 1, size=(N, 3)) # ~U(0, 1)
    gas *= box - np.array((0, 0, sigma * 2))  # ~U(0, box - 2 sigma)
    gas = gas + np.array((0, 0, sigma))  # ~U(sigma, box - sigma)

    N += len(walls)
    configuration = np.concatenate((walls, gas))
    indices_to_move = np.arange(len(walls), N)

else:
    exit("invalid boundary kind: " + kind)

is_pbc = [True, True, False]
is_hard = [False, False, True]
system = hsmc.chard_sphere.HSMC(N, box, is_pbc, is_hard, r_skin)

system.load_positions(configuration.T)
system.set_indices(indices_to_move)
system.fill_hs()
system.crush_along_axis(vf_final, 0.01, 2)

print(system)
print(f"Do particles overlap? {system.report_overlap()}")


trajectory = np.empty((length, N, 3))

for i in range(length):
    for _ in range(jump):
        system.sweep()
    trajectory[i] = system.get_positions().copy().T


time = np.arange(length)
isf = hsmc.analysis.get_isf_3d(
    trajectory, pbc_box=(box[0], box[1], None), length=length
)
popt, pcov = curve_fit(
        f=lambda x, tau, b: np.exp(-(x / tau)**b),
        xdata = time,
        ydata = isf,
        p0 = (10, 1),
)
tau, b = popt
print(f"Relaxation Time: {tau * jump:.4f} sweeps")

if plot_isf:
    plt.scatter(time, isf, marker='o', color='tomato', fc='none', label='data')
    plt.plot(time, np.exp(-(time / tau)**b), color='teal', label='fit')
    plt.text(time[len(time)//2], 0.7, "$\\tau=$" + f"{tau * jump:.0f} sweeps")
    plt.xlabel(f"Lag Time / {jump} sweeps")
    plt.ylabel(f"ISF")
    plt.ylim(-0.1, 1.1)
    plt.tight_layout()
    plt.savefig('isf.pdf')
    if show_isf:
        plt.show()
