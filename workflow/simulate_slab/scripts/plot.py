#!/usr/bin/env python3
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common.workflow_support import (
    active_workflow_uuid,
    box_path,
    density_profile_path,
    figure_path,
    get_workflow_logger,
    inplane_profile_path,
    load_config,
    slab_info_path,
    tcc_bulk_path,
    tcc_spatial_dist_path,
)

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.size"] = 18


def _load_cluster_map(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    names = [str(value) for value in data["cluster_names"]]
    values = data["populations"]
    if values.ndim == 1:
        return {name: float(values[i]) for i, name in enumerate(names)}
    return {name: values[i] for i, name in enumerate(names)}, data


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("plot", current_uuid)
    crystal = conf["Boundary"]["crystal"]
    hkl_raw = conf["Boundary"]["hkl"].split()
    hkl = " ".join(hkl_raw)
    label = f"{crystal} ({hkl})"
    logger.info("Generating plots for %s", label)

    with open(box_path(current_uuid), "r") as f:
        box = json.load(f)
    with open(slab_info_path(current_uuid), "r") as f:
        slab_info = json.load(f)
    z_slab = float(slab_info["z_slab"])

    tcc_dist, spatial_data = _load_cluster_map(tcc_spatial_dist_path(current_uuid))
    tcc_bulk = _load_cluster_map(tcc_bulk_path(current_uuid))
    density_data = np.load(density_profile_path(current_uuid), allow_pickle=False)

    bins = spatial_data["bin_edges"]
    bin_centres = spatial_data["bin_centres"]

    tcc_names = {key: key for key in tcc_dist}
    tcc_names["sp3c"] = "5A"
    tcc_names["sp5c"] = "7A"

    # Density profile
    fig, ax = plt.subplots(1, 1)
    ax.plot(
        density_data["bin_centres"],
        density_data["density"],
        color="tomato",
        label="fluid density",
    )
    ax.axvline(x=0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"Distance to Nearest Crystal Surface / $\sigma$")
    ax.set_ylabel("Number Density")
    ax.set_title(label)
    plt.tight_layout()
    density_plot_path = figure_path("density", ".pdf", current_uuid)
    plt.savefig(density_plot_path)
    plt.close()

    # TCC grid
    cluster_names = ["sp3c", "6A", "sp5c", "8B", "8A", "9B", "10B", "FCC", "HCP"]
    fig, ax = plt.subplots(3, 3)
    fig.suptitle(f"Slab Geometry with {label}")
    ax = ax.ravel()
    for i, key in enumerate(cluster_names):
        ax[i].set_title(tcc_names[key])
        ax[i].plot(
            bin_centres,
            tcc_dist[key],
            color="teal",
            zorder=2,
            label="slab",
        )
        ax[i].plot(
            (bins[0], bins[-1]),
            (tcc_bulk[key], tcc_bulk[key]),
            color="k",
            lw=1,
            ls="--",
            zorder=1,
            label="bulk",
        )
        ax[i].set_ylabel("Population")
        ax[i].set_xlabel(r"Distance to Surface / $\sigma$")
        ax[i].set_xlim(bins[0], bins[-1])

    plt.gcf().set_size_inches(12, 10)
    plt.tight_layout()
    result_1_path = figure_path("tcc_result_1", ".pdf", current_uuid)
    plt.savefig(result_1_path)
    plt.close()

    # Selected clusters on one axis
    fig, ax = plt.subplots(1, 1)
    cluster_names = ["sp5c", "8A", "10B", "FCC", "HCP"]
    for i, key in enumerate(cluster_names):
        if i == 0:
            ax.plot(
                (bins[0], bins[-1]),
                (tcc_bulk[key], tcc_bulk[key]),
                color="k",
                lw=1,
                ls="--",
                zorder=1,
                label="bulk",
            )
        else:
            ax.plot(
                (bins[0], bins[-1]),
                (tcc_bulk[key], tcc_bulk[key]),
                color="k",
                lw=1,
                ls="--",
                zorder=1,
            )
        color = mpl.cm.tab10(i / len(cluster_names))
        ax.plot(
            bin_centres,
            tcc_dist[key],
            zorder=2,
            color=color,
            label=tcc_names[key],
        )
    ax.set_ylabel("Population")
    ax.set_xlabel(r"Distance to Surface / $\sigma$")
    plt.legend(handlelength=1.0, ncol=2, loc="lower center")
    plt.gcf().set_size_inches(8, 5)
    plt.tight_layout()
    result_2_path = figure_path("tcc_result_2", ".pdf", current_uuid)
    plt.savefig(result_2_path)
    plt.close()

    # Deviation from bulk vs distance
    cluster_names = ["sp5c", "8A", "10B", "FCC", "HCP"]
    markers = ["^", "s", "p", "h", "o"]
    fig, ax = plt.subplots(1, 1)
    for i, key in enumerate(cluster_names):
        color = mpl.cm.rainbow(i / len(cluster_names))
        pop = tcc_dist[key]
        val = np.abs(pop - tcc_bulk[key])
        ax.scatter(
            bin_centres,
            val,
            zorder=2,
            color=color,
            ec="k",
            marker=markers[i],
            label=tcc_names[key],
        )
    ax.set_ylabel("|Deviation from Bulk|")
    ax.set_xlabel(r"Distance to Surface / $\sigma$")
    plt.legend(handlelength=1.0, ncol=2, loc="upper right")
    plt.gcf().set_size_inches(8, 5)
    plt.yscale("log")
    plt.tight_layout()
    result_3_path = figure_path("tcc_result_3", ".pdf", current_uuid)
    plt.savefig(result_3_path)
    plt.close()

    # In-plane homogeneity
    inplane_data = np.load(inplane_profile_path(current_uuid), allow_pickle=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(inplane_data["x_bin_centres"], inplane_data["x_density"], color="tomato")
    ax[0].axhline(y=inplane_data["x_density"].mean(), color="k", ls="--", lw=1)
    ax[0].set_xlabel(r"X / $\sigma$")
    ax[0].set_ylabel("Number Density")
    ax[0].set_title("x density")
    ax[1].plot(inplane_data["y_bin_centres"], inplane_data["y_density"], color="tomato")
    ax[1].axhline(y=inplane_data["y_density"].mean(), color="k", ls="--", lw=1)
    ax[1].set_xlabel(r"Y / $\sigma$")
    ax[1].set_ylabel("Number Density")
    ax[1].set_title("y density")
    plt.tight_layout()
    inplane_plot_path = figure_path("inplane_density", ".pdf", current_uuid)
    plt.savefig(inplane_plot_path)
    plt.close()

    logger.info(
        "Saved plots: %s, %s, %s, %s, %s",
        density_plot_path.name,
        inplane_plot_path.name,
        result_1_path.name,
        result_2_path.name,
        result_3_path.name,
    )


if __name__ == "__main__":
    main()
