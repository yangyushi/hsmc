#!/usr/bin/env python3
import os
import json
import numpy as np
import configparser
import matplotlib.pyplot as plt

import hsmc


conf = configparser.ConfigParser()
conf.read('configure.ini')

valid_planes = ('fcc100', 'fcc111', 'fcc110')

# load parameters
N = int(conf['System']['n'])
sigma = 1
vf_init = float(conf['System']['vf_init'])
vf_final = float(conf['System']['vf_final'])
vf_crystal = float(conf['Boundary']['vf_crystal'])
z_final = float(conf['Boundary']['z'])
kind = conf['Boundary']['kind']
sweep_equilibrium = int(float(conf['Run']['equilibrium']))
sweep_total = int(float(conf['Run']['total']))
dump_frequency = int(float(conf['Run']['dump_frequency']))
dump_name = os.path.join('result', conf['Run']['filename'])

v_sph = np.pi * N * sigma ** 3 / 6
l_xy = np.sqrt(v_sph / vf_final / z_final)  # approx value

# setup the boundary
if kind == 'hardwall':
    z_init = v_sph / vf_init / l_xy ** 2
    box = np.array((l_xy, l_xy, z_init))
    configuration = np.random.uniform(0, 1, (N, 3)) * box
    indices_to_move = np.arange(N)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*configuration.T)
    plt.tight_layout()
    plt.savefig("system_start.png")

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

    v_sph = np.pi * N * sigma ** 3 / 6
    a_xy = box_xy[0] * box_xy[1]  # area of box in xy plane
    box_z = v_sph / a_xy / vf_init

    # (x, y) --> (x, y, 0)
    plane = np.concatenate((plane, np.zeros((len(plane), 1))), axis=1)

    # concatenate top and bottom planes
    walls = np.concatenate((plane, plane + np.array((0, 0, box_z))), axis=0)

    box = np.array((box_xy[0], box_xy[1], box_z))

    gas = np.random.uniform(0, 1, size=(N - len(walls), 3)) # ~U(0, 1)
    gas *= box - np.array((0, 0, sigma * 2))  # ~U(0, box - 2 sigma)
    gas = gas + np.array((0, 0, sigma))  # ~U(sigma, box - sigma)

    configuration = np.concatenate((walls, gas))
    indices_to_move = np.arange(len(walls), N)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(*walls.T)
    ax.scatter(*gas.T)
    plt.tight_layout()
    plt.savefig("system_start.png")
else:
    exit("invalid boundary kind: " + kind)

is_pbc = [True, True, False]
is_hard = [False, False, True]
system = hsmc.chard_sphere.HSMC(N, box, is_pbc, is_hard)

system.load_positions(configuration.T)
system.set_indices(indices_to_move)
system.fill_hs()
system.crush_along_axis(vf_final, 0.01, 2)

print(system)
print(f"Do particles overlap? {system.report_overlap()}")


pos = system.copy_positions()
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
ax.scatter(*pos, color='w', ec='k', marker='o')
plt.title(f"{kind}, $\phi$ = {vf_final}")
ax.set_xlabel("X / $\sigma$")
ax.set_ylabel("Y / $\sigma$")
ax.set_zlabel("Z / $\sigma$")
ax2.hist(pos[2], bins=250)
ax2.set_xlabel("Z / $\sigma$")
ax2.set_ylabel("PDF")
plt.tight_layout()
plt.savefig("system_crushed.png")


for _ in range(sweep_equilibrium):
    system.sweep()

f_xyz = open(dump_name, 'w')
f_xyz.close()

f_xyz = open(dump_name, 'a')

n_move = len(indices_to_move)
for frame in range(sweep_total):
    system.sweep()
    if frame % dump_frequency == 0:
        tmp = system.copy_positions().T[indices_to_move]
        np.savetxt(
            f_xyz, tmp, delimiter=' ',
            fmt=['A %.8e'] + ['%.8e' for i in range(2)],
            header='%s\nframe %s' % (n_move, frame),
            comments='',
        )

f_xyz.close()

with open(os.path.join("result", "box.json"), 'w') as f:
    json.dump(system.get_box(), f)
