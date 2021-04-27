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
from tcc_python_scripts.tcc import wrapper


# Load Parameters
conf = configparser.ConfigParser()
conf.read('configure.ini')
n_particle = int(conf['System']['n'])
vf_init = float(conf['System']['vf_init'])
sweep_equilibrium = int(float(conf['Run']['equilibrium']))
sweep_total = int(float(conf['Run']['total']))
dump_frequency_bulk = int(float(conf['Run']['dump_frequency_bulk']))
nbins = int(conf['Analyse']['nbins'])
kind = conf['Boundary']['kind']

with open(os.path.join("result", "box.json"), "r") as f:
    box = json.load(f)

fake_box = [l for l in box]
fake_box[-1] *= 10  # to remove PBC for TCC

# Load Data
frames = hsmc.analysis.XYZ(
    os.path.join("result", conf['Run']['filename']),
    delimiter=' ', usecols=[1, 2, 3],
    engine='pandas'
)


# Density Profile
be = np.linspace(0, box[-1], 1000)
bc = (be[1:] + be[:-1]) / 2
dz = be[1] - be[0]
v_bin = dz * box[0] * box[1]

z = []
for i, frame in enumerate(frames):
    z.append(frame[:, -1])

z = np.concatenate(z)
z_mid = z.mean()
hist, _ = np.histogram(z, bins=be)

plt.plot((z_mid, z_mid), (0, 2), color='k', lw=1)
plt.plot((0, box[2]), (0, 0), color='k', lw=1)

plt.plot(bc, hist / len(frames) / v_bin, color='tomato')
plt.xlabel('Z / $\sigma$')
plt.ylabel('Numble Density')
plt.tight_layout()
plt.savefig('density.pdf')
plt.close()


bulk_vf = hsmc.analysis.get_bulk_vf(frames, box, jump=1, npoints=50)

if "tcc_bulk.pkl" in os.listdir('result'):
    bulk = pd.read_pickle(os.path.join('result', 'tcc_bulk.pkl'))['Mean Pop Per Frame']
else:  # find tcc populations in equivalent bulk system
    L = (np.pi * n_particle / 6 / vf_init) ** (1.0 / 3.0)
    box = [L, L, L]

    is_pbc = [True, True, True]
    is_hard = [False, False, False]
    system = hsmc.chard_sphere.HSMC(n_particle, box, is_pbc, is_hard)
    system.fill_hs()
    system.crush(bulk_vf, 0.01)
    dump_name = os.path.join("result", "sample_bulk.xyz")

    print(system)
    print(f"Do particles overlap in Bulk? {system.report_overlap()}")

    for _ in range(sweep_equilibrium):
        system.sweep()

    f_xyz = open(dump_name, 'w')
    f_xyz.close()
    f_xyz = open(dump_name, 'a')

    for frame in range(sweep_total):
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

    box_bulk = system.get_box()
    frames_bulk = hsmc.analysis.XYZ(
        dump_name, delimiter=' ', usecols=[1, 2, 3],
        engine='pandas'
    )
    tcc_bulk = wrapper.TCCWrapper()
    tcc_bulk.input_parameters['Simulation']['voronoi_parameter'] = 0.82
    tcc_bulk.input_parameters['Simulation']['rcutAA'] = 2.0
    tcc_bulk.input_parameters['Output']['Raw'] = True
    tcc_results_bulk = tcc_bulk.run(box_bulk, frames_bulk[::1])
    tcc_results_bulk.to_pickle(os.path.join('result', 'tcc_bulk.pkl'))
    bulk = tcc_results_bulk['Mean Pop Per Frame']


if "tcc_dict_slit.pkl" in os.listdir('result'):
    with open(os.join("result", "tcc_dict_slit.pkl"), "rb") as f:
        tcc_dicts = pickle.load(f)
else:  # perform TCC analysis
    tcc = wrapper.TCCWrapper()
    tcc.input_parameters['Simulation']['voronoi_parameter'] = 0.82
    tcc.input_parameters['Simulation']['rcutAA'] = 2.0
    tcc.input_parameters['Output']['Raw'] = True
    results_ = tcc.run(fake_box, frames[::1], output_directory='./tcc')
    tcc_dicts = tcc.get_cluster_dict()
    with open('tcc_dict_slit.pkl', 'wb') as f:
        pickle.dump(tcc_dicts, f)


#first analysis
n_frames = len(frames)

bins = np.linspace(-z_mid, z_mid, nbins)
bc = (bins[1:] + bins[:-1]) / 2

fig, ax = plt.subplots(3, 3)
fig.suptitle(f"Slit Geometry with {kind}")
ax = ax.ravel()

tcc_names = {key:key for key in tcc_dicts}
tcc_names['sp3c'] = '5A'
tcc_names['sp5c'] = '7A'

for i, key in enumerate(['sp3c', '6A', 'sp5c', '8B', '8A', '9B', '10B', 'FCC', 'HCP']):
    count = np.concatenate(tcc_dicts[key], axis=0).ravel().astype(int)
    stat_tcc, _, _ = binned_statistic( x=z-z_mid, values=count, statistic='sum', bins=bins )
    stat_all, _, _ = binned_statistic( x=z-z_mid, values=np.ones(count.shape), statistic='sum', bins=bins )
    stat_all[stat_all == 0] = np.nan
    ax[i].set_title(tcc_names[key])
    ax[i].plot(
        (bins[0], bins[-1]), (bulk[key], bulk[key]), color='k',
        lw=1, ls='--', zorder=1, label='bulk'
    )
    ax[i].plot(
        bc, stat_tcc / stat_all, color='teal', zorder=2, label="Slit with (100) Facet"
    )
    ax[i].set_ylabel('Population')
    ax[i].set_xlabel('Z / $\sigma$')
    ax[i].set_xlim(bc[0], bc[-1])

plt.gcf().set_size_inches(12, 10)
plt.tight_layout()
plt.savefig('tcc_result_1.pdf')
plt.close()

# second analysis

bins = np.linspace(-z_mid, z_mid, nbins)
bc = (bins[1:] + bins[:-1]) / 2

fig, ax = plt.subplots(1, 1)
clusters_to_plot = ['sp5c', '8A', '10B', 'FCC', 'HCP']

for i, key in enumerate(clusters_to_plot):
    count = np.concatenate(tcc_dicts[key], axis=0).ravel().astype(int)
    stat_tcc, _, _ = binned_statistic( x=z-z_mid, values=count, statistic='sum', bins=bins )
    stat_all, _, _ = binned_statistic( x=z-z_mid, values=np.ones(count.shape), statistic='sum', bins=bins )
    stat_all[stat_all == 0] = np.nan
    if i == 0:
        ax.plot(
            (bins[0], bins[-1]), (bulk[key], bulk[key]), color='k',
            lw=1, ls='--', zorder=1, label='bulk'
        )
    else:
        ax.plot(
            (bins[0], bins[-1]), (bulk[key], bulk[key]), color='k',
            lw=1, ls='--', zorder=1
        )
    color = mpl.cm.tab10(i / len(clusters_to_plot))
    ax.plot(
        bc, stat_tcc / stat_all, zorder=2, color=color,
        label=tcc_names[key]
    )
ax.set_ylabel('Population')
ax.set_xlabel('Z / $\sigma$')
ax.set_xlim(-z_mid, z_mid)
plt.legend(handlelength=1.0, ncol=2, loc='lower center')

plt.gcf().set_size_inches(8, 5)
plt.tight_layout()
plt.savefig('tcc_result_2.pdf')
plt.close()

# third analysis

bins = np.linspace(-z_mid, z_mid, nbins)
bc = (bins[1:] + bins[:-1]) / 2

fig, ax = plt.subplots(1, 1)
clusters_to_plot = ['sp5c', '8A', '10B', 'FCC', 'HCP']

for i, key in enumerate(clusters_to_plot):
    count = np.concatenate(tcc_dicts[key], axis=0).ravel().astype(int)
    stat_tcc, _, _ = binned_statistic( x=z-z_mid, values=count, statistic='sum', bins=bins )
    stat_all, _, _ = binned_statistic( x=z-z_mid, values=np.ones(count.shape), statistic='sum', bins=bins )
    stat_all[stat_all == 0] = np.nan
    color = mpl.cm.tab10(i / len(clusters_to_plot))
    pop = stat_tcc / stat_all
    val = np.abs(pop - bulk[key])
    ax.plot(
        bc, val, zorder=2, color=color,
        label=tcc_names[key]
    )
ax.set_ylabel('|Deviation from Bulk|')
ax.set_xlabel('Z / $\sigma$')
ax.set_xlim(-z_mid, z_mid)
plt.legend(handlelength=1.0, ncol=2, loc='lower center')

plt.gcf().set_size_inches(8, 5)
plt.yscale('log')
plt.tight_layout()
plt.savefig('tcc_result_3.pdf')
plt.close()
