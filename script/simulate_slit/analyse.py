import os
import json
import pickle
import numpy as np
import pandas as pd
import configparser
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
mpl.rcParams['font.size'] = 18

import hsmc


# Load Parameters
conf = configparser.ConfigParser()
conf.read('configure.ini')
n_particle = int(conf['System']['n'])
vf_init = float(conf['System']['vf_init'])
sweep_equilibrium = int(float(conf['Run']['equilibrium']))
sweep_total_bulk = int(float(conf['Run']['total_bulk']))
dump_frequency_bulk = int(float(conf['Run']['dump_frequency_bulk']))
nbins = int(conf['Analyse']['nbins'])
kind = conf['Boundary']['kind']
dump_name = os.path.join("result", conf['Run']['filename'])

with open(os.path.join("result", "box.json"), "r") as f:
    box = json.load(f)

fake_box = [l for l in box]
fake_box[-1] *= 10  # to remove PBC for TCC

# Load Data
frames = hsmc.analysis.XYZ(
    dump_name, delimiter=' ', usecols=[1, 2, 3], engine='pandas'
)


# Density Profile
be = np.linspace(0, box[-1], 1000)
bc = (be[1:] + be[:-1]) / 2
dz = be[1] - be[0]
v_bin = dz * box[0] * box[1]

hist = np.zeros(bc.shape)
z_mid = 0
for i, frame in enumerate(frames):
    z = frame[:, 2]
    hist += np.histogram(z, bins=be)[0]
    z_mid += z.mean()
z_mid /= len(frames)

plt.plot((z_mid, z_mid), (0, 2), color='k', lw=1)
plt.plot((0, box[2]), (0, 0), color='k', lw=1)

plt.plot(bc, hist / len(frames) / v_bin, color='tomato')
plt.xlabel('Z / $\sigma$')
plt.ylabel('Numble Density')
plt.tight_layout()
plt.savefig('density.pdf')
plt.close()


bulk_vf = hsmc.analysis.get_bulk_vf(frames, box, jump=1, npoints=50)

L = (np.pi * n_particle / 6 / vf_init) ** (1.0 / 3.0)
box = [L, L, L]

is_pbc = [True, True, True]
is_hard = [False, False, False]
system = hsmc.chard_sphere.HSMC(n_particle, box, is_pbc, is_hard)
system.fill_hs()
system.crush(bulk_vf, 0.01)
dump_name_bulk = os.path.join("result", "sample_bulk.xyz")

print(system)
print(f"Do particles overlap in Bulk? {system.report_overlap()}")

for _ in range(sweep_equilibrium):
    system.sweep()

f_xyz = open(dump_name_bulk, 'w')
f_xyz.close()
f_xyz = open(dump_name_bulk, 'a')

for frame in range(sweep_total_bulk):
    system.sweep()
    if frame % dump_frequency_bulk == 0:
        tmp = system.copy_positions().T
        np.savetxt(
            f_xyz, tmp, delimiter=' ',
            fmt=['A %.8e'] + ['%.8e' for i in range(2)],
            header='%s\nframe %s' % (n_particle, frame),
            comments='',
        )

f_xyz.close()

tcc_parameters = {'voronoi_parameter':0.82, 'rcutAA': 2.0}

box_bulk = system.get_box()
frames_bulk = hsmc.analysis.XYZ(
    dump_name_bulk, delimiter=' ', usecols=[1, 2, 3], engine='pandas'
)
tcc_bulk = hsmc.TCC('tcc_bulk')
tcc_bulk.run(
    os.path.join('..', dump_name_bulk),
    box_bulk, Raw=False, **tcc_parameters
)
with open(os.path.join('result', 'tcc_bulk.pkl'), 'wb') as f:
    pickle.dump(tcc_bulk.population.mean(axis=0), f)


tcc = hsmc.TCC('tcc')
tcc.run(
    os.path.join('..', dump_name),
    fake_box, Raw=True, **tcc_parameters
)
tcc.parse()



bins = np.linspace(-z_mid, z_mid, nbins)
bc = (bins[1:] + bins[:-1]) / 2


tcc_spatial_dist = {}
for cn in tcc.clusters:
    stat_tcc = np.zeros(bc.shape)
    stat_all = np.zeros(bc.shape)
    for f in range(len(tcc)):
        count = tcc.clusters[cn][f].ravel().astype(int)
        z = frames[f][:, 2]
        stat_tcc += binned_statistic(
            x=z-z_mid, values=count, statistic='sum', bins=bins
        )[0]
        stat_all += binned_statistic(
            x=z-z_mid, values=np.ones(count.shape), statistic='sum', bins=bins
        )[0]
    stat_all[stat_all == 0] = np.nan
    tcc_spatial_dist.update({
        cn : stat_tcc / stat_all
    })

with open(os.path.join('result', 'tcc_spatial_dist.pkl'), 'wb') as f:
    pickle.dump(
        {
            'bin_edges': bins,
            'bin_centres': bc,
            'hist': tcc_spatial_dist,
        },
        f
    )
