#!/usr/bin/env python3
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from common.workflow_support import (
    box_path,
    figure_path,
    load_config,
    tcc_bulk_path,
    tcc_spatial_dist_path,
    workflow_uuid,
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
    current_uuid = workflow_uuid()
    kind = conf["Boundary"]["kind"]

    tcc_dist, spatial_data = _load_cluster_map(tcc_spatial_dist_path(current_uuid))
    tcc_bulk = _load_cluster_map(tcc_bulk_path(current_uuid))
    with open(box_path(current_uuid), "r") as f:
        box = json.load(f)

    bins = spatial_data["bin_edges"]
    bin_centres = spatial_data["bin_centres"]

    tcc_names = {key: key for key in tcc_dist}
    tcc_names["sp3c"] = "5A"
    tcc_names["sp5c"] = "7A"

    cluster_names = ["sp3c", "6A", "sp5c", "8B", "8A", "9B", "10B", "FCC", "HCP"]
    fig, ax = plt.subplots(3, 3)
    fig.suptitle(f"Slit Geometry with {kind}")
    ax = ax.ravel()
    for i, key in enumerate(cluster_names):
        ax[i].set_title(tcc_names[key])
        ax[i].plot(
            bin_centres,
            tcc_dist[key],
            color="teal",
            zorder=2,
            label="Slit with (100) Facet",
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
        ax[i].set_xlabel(r"Z / $\sigma$")
        ax[i].set_xlim(bins[0], bins[-1])

    plt.gcf().set_size_inches(12, 10)
    plt.tight_layout()
    plt.savefig(figure_path("tcc_result_1", ".pdf", current_uuid))
    plt.close()

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
        color = mpl.cm.tab10(np.random.random())
        ax.plot(bin_centres, tcc_dist[key], zorder=2, color=color, label=tcc_names[key])
    ax.set_ylabel("Population")
    ax.set_xlabel(r"Z / $\sigma$")
    plt.legend(handlelength=1.0, ncol=2, loc="lower center")
    plt.gcf().set_size_inches(8, 5)
    plt.tight_layout()
    plt.savefig(figure_path("tcc_result_2", ".pdf", current_uuid))
    plt.close()

    cluster_names = ["sp5c", "8A", "10B", "FCC", "HCP"]
    markers = ["^", "s", "p", "h", "o"]
    fig, ax = plt.subplots(1, 1)
    for i, key in enumerate(cluster_names):
        color = mpl.cm.rainbow(np.random.random())
        pop = tcc_dist[key]
        val = np.abs(pop - tcc_bulk[key])
        ax.scatter(
            box[-1] / 2 - np.abs(bin_centres),
            val,
            zorder=2,
            color=color,
            ec="k",
            marker=markers[i],
            label=tcc_names[key],
        )
    ax.set_ylabel("|Deviation from Bulk|")
    ax.set_xlabel(r"Distance to Wall / $\sigma$")
    plt.legend(handlelength=1.0, ncol=2, loc="upper right")
    plt.gcf().set_size_inches(8, 5)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(figure_path("tcc_result_3", ".pdf", current_uuid))
    plt.close()


if __name__ == "__main__":
    main()
