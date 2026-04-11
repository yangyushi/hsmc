import os
import json
import pickle
import numpy as np
import configparser
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Arial'
mpl.rcParams['font.size'] = 18


conf = configparser.ConfigParser()
conf.read('configure.ini')
kind = conf['Boundary']['kind']

path_tcc = os.path.join('result', 'tcc_spatial_dist.pkl')
path_bulk = os.path.join('result', 'tcc_bulk.pkl')
path_box = os.path.join("result", "box.json")

with open(path_tcc, 'rb') as f:
    data = pickle.load(f)
    tcc_dist = data['hist']
    bins = data['bin_edges']
    bin_centres = data['bin_centres']

tcc_names = {key:key for key in tcc_dist}
tcc_names['sp3c'] = '5A'
tcc_names['sp5c'] = '7A'

with open(path_bulk, 'rb') as f:
    tcc_bulk = pickle.load(f)

with open(path_box, "r") as f:
    box = json.load(f)

#first analysis
cluster_names = ['sp3c', '6A', 'sp5c', '8B', '8A', '9B', '10B', 'FCC', 'HCP']
fig, ax = plt.subplots(3, 3)
fig.suptitle(f"Slit Geometry with {kind}")
ax = ax.ravel()
for i, key in enumerate(cluster_names):
    ax[i].set_title(tcc_names[key])
    ax[i].plot(
        bin_centres, tcc_dist[key], color='teal', zorder=2,
        label="Slit with (100) Facet"
    )

    ax[i].plot(
        (bins[0], bins[-1]), (tcc_bulk[key], tcc_bulk[key]), color='k',
        lw=1, ls='--', zorder=1, label='bulk'
    )
    ax[i].set_ylabel('Population')
    ax[i].set_xlabel('Z / $\sigma$')
    ax[i].set_xlim(bins[0], bins[-1])

plt.gcf().set_size_inches(12, 10)
plt.tight_layout()
plt.savefig('figure/tcc_result_1.pdf')
plt.close()

# second analysis
fig, ax = plt.subplots(1, 1)
cluster_names = ['sp5c', '8A', '10B', 'FCC', 'HCP']

for i, key in enumerate(cluster_names):
    if i == 0:
        ax.plot(
            (bins[0], bins[-1]), (tcc_bulk[key], tcc_bulk[key]), color='k',
            lw=1, ls='--', zorder=1, label='bulk'
        )
    else:
        ax.plot(
            (bins[0], bins[-1]), (tcc_bulk[key], tcc_bulk[key]), color='k',
            lw=1, ls='--', zorder=1
        )
    color = mpl.cm.tab10(np.random.random())
    ax.plot(
        bin_centres, tcc_dist[key], zorder=2, color=color,
        label=tcc_names[key]
    )
ax.set_ylabel('Population')
ax.set_xlabel('Z / $\sigma$')
plt.legend(handlelength=1.0, ncol=2, loc='lower center')

plt.gcf().set_size_inches(8, 5)
plt.tight_layout()
plt.savefig('figure/tcc_result_2.pdf')
plt.close()

# third analysis
cluster_names = ['sp5c', '8A', '10B', 'FCC', 'HCP']
markers = ['^', 's', 'p', 'h', 'o']

fig, ax = plt.subplots(1, 1)
for i, key in enumerate(cluster_names):
    color = mpl.cm.rainbow(np.random.random())
    pop = tcc_dist[key]
    val = np.abs(pop - tcc_bulk[key])
    ax.scatter(
        box[-1] / 2 - np.abs(bin_centres), val,
        zorder=2, color=color, ec='k', marker=markers[i],
        label=tcc_names[key]
    )
ax.set_ylabel('|Deviation from Bulk|')
ax.set_xlabel('Distance to Wall / $\sigma$')
plt.legend(handlelength=1.0, ncol=2, loc='upper right')

plt.gcf().set_size_inches(8, 5)
plt.yscale('log')
plt.tight_layout()
plt.savefig('figure/tcc_result_3.pdf')
plt.close()
