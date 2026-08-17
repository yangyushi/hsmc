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
    density_profile_path,
    get_workflow_logger,
    inplane_profile_path,
    load_config,
    slab_info_path,
    slit_sample_path,
    tcc_spatial_dist_path,
)

mpl.rcParams["font.size"] = 18


def _distance_to_nearest_surface(z: np.ndarray, z_slab: float, z_final: float) -> np.ndarray:
    """Distance from each fluid particle to the nearest crystal surface.

    The slab is centred in the box, occupying
    [z_final/2 - z_slab/2, z_final/2 + z_slab/2]. Periodic images of the two
    surfaces are also considered.
    """
    z_c = z_final / 2.0
    z1 = z_c - z_slab / 2.0
    z2 = z_c + z_slab / 2.0

    def pbc_distance(z, z_surf):
        d = np.abs(z - z_surf)
        d = np.minimum(d, np.abs(z - z_surf - z_final))
        d = np.minimum(d, np.abs(z - z_surf + z_final))
        return d

    return np.minimum(pbc_distance(z, z1), pbc_distance(z, z2))


def _density_profile(frames, box, z_slab, nbins, n_crystal=0, boundary_thickness=0.0):
    Lx, Ly, z_final = box
    max_d = (z_final - z_slab) / 2.0
    be = np.linspace(0.0, max_d, nbins + 1)
    bc = (be[1:] + be[:-1]) / 2.0
    hist = np.zeros(nbins)
    for frame in frames:
        frame = frame[n_crystal:]
        d = _distance_to_nearest_surface(frame[:, 2], z_slab, z_final)
        hist += np.histogram(d, bins=be)[0]
    # Two crystal-fluid interfaces contribute to the same distance bin.
    density = hist / (len(frames) * 2.0 * Lx * Ly * (be[1:] - be[:-1]))

    # Bulk number density in the region far from the crystal surfaces.
    bulk_mask = bc > boundary_thickness
    if np.any(bulk_mask):
        bulk_density = float(np.mean(density[bulk_mask]))
    else:
        bulk_density = float(np.nan)
    bulk_vf = bulk_density * np.pi / 6.0

    return bc, density, bulk_density, bulk_vf


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("analysis", current_uuid)

    dump_name = slit_sample_path(conf, current_uuid)
    frames = hsmc.analysis.XYZ(
        dump_name,
        delimiter=" ",
        usecols=[1, 2, 3],
        engine="pandas",
    )

    with open(box_path(current_uuid), "r") as f:
        box = np.array(json.load(f))
    with open(slab_info_path(current_uuid), "r") as f:
        slab_info = json.load(f)
    z_slab = float(slab_info["z_slab"])
    n_crystal = int(slab_info["N_crystal"])
    boundary_thickness = float(slab_info.get("boundary_thickness", 0.0))

    nbins = int(conf["Analyse"]["nbins"])
    if nbins == 0:
        nbins = max(1200, int(box[-1] * 24))

    # Density profile
    logger.info("Building density profile")
    density_path = density_profile_path(current_uuid)
    bc, density, bulk_density, bulk_vf = _density_profile(
        frames, box, z_slab, nbins, n_crystal, boundary_thickness
    )
    np.savez(
        density_path,
        bin_centres=bc,
        density=density,
        bulk_density=bulk_density,
        bulk_vf=bulk_vf,
        boundary_thickness=boundary_thickness,
    )
    logger.info(
        "Saved density profile: %s (bulk density=%.6f, bulk vf=%.6f)",
        density_path.name,
        bulk_density,
        bulk_vf,
    )

    # In-plane homogeneity check
    logger.info("Building in-plane density profiles")
    x_edges = np.linspace(0.0, box[0], 51)
    y_edges = np.linspace(0.0, box[1], 51)
    x_hist = np.zeros(50)
    y_hist = np.zeros(50)
    for frame in frames:
        fluid = frame[n_crystal:]
        x_hist += np.histogram(fluid[:, 0], bins=x_edges)[0]
        y_hist += np.histogram(fluid[:, 1], bins=y_edges)[0]
    x_hist = x_hist / (len(frames) * box[1] * (x_edges[1] - x_edges[0]))
    y_hist = y_hist / (len(frames) * box[0] * (y_edges[1] - y_edges[0]))
    np.savez(
        inplane_profile_path(current_uuid),
        x_bin_centres=(x_edges[1:] + x_edges[:-1]) / 2.0,
        y_bin_centres=(y_edges[1:] + y_edges[:-1]) / 2.0,
        x_density=x_hist,
        y_density=y_hist,
    )
    logger.info("Saved in-plane density profiles")

    # Spatial TCC distribution
    output_path = tcc_spatial_dist_path(current_uuid)
    if output_path.is_file():
        logger.info("Reusing existing output: %s", output_path.name)
        return

    tcc_parameters = {
        "voronoi_parameter": 0.82,
        "rcutAA": 2.0,
    }

    logger.info("Running TCC on fluid particles")
    fluid_frames = np.array(frames[:])[:, n_crystal:]
    tcc_parser = tcc.OTF()
    tcc_parser.run(fluid_frames, box, **tcc_parameters)
    tcc_parser.parse()

    logger.info("Calculating spatial TCC statistics")
    max_d = (box[-1] - z_slab) / 2.0
    bins = np.linspace(0.0, max_d, nbins + 1)
    bin_centres = (bins[1:] + bins[:-1]) / 2.0

    cluster_names = list(tcc_parser.cluster_bool.keys())
    populations = np.zeros((len(cluster_names), len(bin_centres)))
    for idx, cluster_name in enumerate(cluster_names):
        stat_tcc = np.zeros(bin_centres.shape)
        stat_all = np.zeros(bin_centres.shape)
        for frame_index in range(len(tcc_parser)):
            count = tcc_parser.cluster_bool[
                cluster_name
            ][frame_index].ravel().astype(int)
            d = _distance_to_nearest_surface(
                fluid_frames[frame_index][:, 2], z_slab, box[-1]
            )
            stat_tcc += binned_statistic(
                x=d, values=count, statistic="sum", bins=bins
            )[0]
            stat_all += binned_statistic(
                x=d,
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
