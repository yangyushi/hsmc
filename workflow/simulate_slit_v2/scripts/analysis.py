#!/usr/bin/env python3
import json

import matplotlib as mpl
import numpy as np
import tcc
from scipy.stats import binned_statistic

import hsmc
from common.workflow_support import (
    active_workflow_uuid,
    box_path,
    get_workflow_logger,
    load_config,
    slit_sample_path,
    tcc_spatial_dist_path,
)

mpl.rcParams["font.size"] = 18


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("analysis", current_uuid)
    output_path = tcc_spatial_dist_path(current_uuid)
    if output_path.is_file():
        logger.info("Reusing existing output: %s", output_path.name)
        return

    dump_name = slit_sample_path(conf, current_uuid)
    frames = hsmc.analysis.XYZ(
        dump_name,
        delimiter=" ",
        usecols=[1, 2, 3],
        engine="pandas",
    )

    nbins = int(conf["Analyse"]["nbins"])
    tcc_parameters = {"voronoi_parameter": 0.82, "rcutAA": 2.0}

    with open(box_path(current_uuid), "r") as f:
        box = np.array(json.load(f))
    fake_box = box.copy()
    fake_box[-1] += tcc_parameters["rcutAA"]

    be = np.linspace(0, box[-1], 1000)
    hist = np.zeros(be.shape[0] - 1)
    z_mid = 0
    for frame in frames:
        z = frame[:, 2]
        hist += np.histogram(z, bins=be)[0]
        z_mid += z.mean()
    z_mid /= len(frames)

    logger.info("Running TCC")
    tcc_parser = tcc.Parser("tcc")
    tcc_parser.run(dump_name, fake_box, Raw=True, clusts=True, **tcc_parameters)
    tcc_parser.parse()

    logger.info("Calculating statistics")
    bins = np.linspace(-z_mid, z_mid, nbins)
    bin_centres = (bins[1:] + bins[:-1]) / 2

    cluster_names = list(tcc_parser.cluster_bool.keys())
    populations = np.zeros((len(cluster_names), len(bin_centres)))
    for idx, cluster_name in enumerate(cluster_names):
        stat_tcc = np.zeros(bin_centres.shape)
        stat_all = np.zeros(bin_centres.shape)
        for frame_index in range(len(tcc_parser)):
            count = tcc_parser.cluster_bool[
                cluster_name
            ][frame_index].ravel().astype(int)
            z = frames[frame_index][:, 2]
            stat_tcc += binned_statistic(
                x=z - z_mid, values=count, statistic="sum", bins=bins
            )[0]
            stat_all += binned_statistic(
                x=z - z_mid,
                values=np.ones(count.shape),
                statistic="sum",
                bins=bins,
            )[0]
        stat_all[stat_all == 0] = np.nan
        populations[idx] = stat_tcc / stat_all

    np.savez(
        output_path,
        cluster_names=np.array(cluster_names),
        bin_edges=bins,
        bin_centres=bin_centres,
        populations=populations,
    )
    logger.info("Saved TCC spatial distribution: %s", output_path.name)


if __name__ == "__main__":
    main()
