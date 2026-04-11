#!/usr/bin/env python3
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
import tcc


conf = configparser.ConfigParser()
conf.read('configure.ini')

dump_name = os.path.join("result", conf['Run']['filename'])

# Load Data
frames = hsmc.analysis.XYZ(
    dump_name, delimiter=' ', usecols=[1, 2, 3], engine='pandas'
)

nbins = int(conf['Analyse']['nbins'])
tcc_parameters = {'voronoi_parameter':0.82, 'rcutAA': 2.0}

with open(os.path.join("result", "box.json"), "r") as f:
    box = np.array(json.load(f))
fake_box = box.copy()
fake_box[-1] += tcc_parameters['rcutAA']

be = np.linspace(0, box[-1], 1000)
bc = (be[1:] + be[:-1]) / 2
hist = np.zeros(bc.shape)
z_mid = 0

for i, frame in enumerate(frames):
    z = frame[:, 2]
    hist += np.histogram(z, bins=be)[0]
    z_mid += z.mean()
z_mid /= len(frames)

print("Running the TCC")
tcc_parser = tcc.Parser('tcc')
tcc_parser.run(
    dump_name, fake_box,
    Raw=True, clusts=True,
    **tcc_parameters,
)
tcc_parser.parse()

print("Calculate the Statistics")

bins = np.linspace(-z_mid, z_mid, nbins)
bc = (bins[1:] + bins[:-1]) / 2

tcc_spatial_dist = {}
for cn in tcc_parser.cluster_bool:
    stat_tcc = np.zeros(bc.shape)
    stat_all = np.zeros(bc.shape)
    for f in range(len(tcc_parser)):
        count = tcc_parser.cluster_bool[cn][f].ravel().astype(int)
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
