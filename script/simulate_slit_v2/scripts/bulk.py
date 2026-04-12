#!/usr/bin/env python3
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tcc

import hsmc
from common.workflow_support import (
    bulk_box_path,
    bulk_sample_path,
    box_path,
    cluster_population_mapping,
    density_profile_path,
    ensure_workflow_metadata,
    figure_path,
    load_config,
    resolve_dump_frequency,
    slit_sample_path,
    tcc_bulk_path,
    write_workflow_log,
)

mpl.rcParams["font.size"] = 18


def main():
    conf = load_config()
    metadata = ensure_workflow_metadata()
    current_uuid = metadata["workflow_uuid"]

    n_particle = int(conf["System"]["n"])
    vf_init = float(conf["System"]["vf_init"])
    r_skin = float(conf["System"]["r_skin"])
    sweep_equilibrium = int(float(conf["Run"]["equilibrium"]))
    sweep_total_bulk = int(float(conf["Run"]["total_bulk"]))
    dump_frequency_bulk_config = conf["Run"]["dump_frequency_bulk"]

    dump_frequency_bulk = resolve_dump_frequency(dump_frequency_bulk_config, current_uuid)
    write_workflow_log(
        "resolved_dump_frequency",
        "bulk",
        current_uuid=current_uuid,
        configured_value=dump_frequency_bulk_config,
        dump_frequency=dump_frequency_bulk,
    )

    slit_box_path = box_path(current_uuid)
    with open(slit_box_path, "r") as f:
        box = json.load(f)

    slit_frames = hsmc.analysis.XYZ(
        slit_sample_path(conf, current_uuid),
        delimiter=" ",
        usecols=[1, 2, 3],
        engine="pandas",
    )

    print("Making the density profile")
    density_path = density_profile_path(current_uuid)
    if not density_path.is_file():
        be = np.linspace(0, box[-1], 1000)
        bc = (be[1:] + be[:-1]) / 2
        dz = be[1] - be[0]
        v_bin = dz * box[0] * box[1]

        hist = np.zeros(bc.shape)
        z_mid = 0
        for frame in slit_frames:
            z = frame[:, 2]
            hist += np.histogram(z, bins=be)[0]
            z_mid += z.mean()
        z_mid /= len(slit_frames)
        density = hist / len(slit_frames) / v_bin

        np.savez(density_path, bin_centres=bc, density=density, z_mid=np.array([z_mid]))
        plt.plot((z_mid, z_mid), (0, 2), color="k", lw=1)
        plt.plot((0, box[2]), (0, 0), color="k", lw=1)
        plt.plot(bc, density, color="tomato")
        plt.xlabel(r"Z / $\sigma$")
        plt.ylabel("Numble Density")
        plt.tight_layout()
        plt.savefig(figure_path("density", ".pdf", current_uuid))
        plt.close()

    sample_bulk_path = bulk_sample_path(current_uuid)
    current_bulk_box_path = bulk_box_path(current_uuid)
    if not sample_bulk_path.is_file() or not current_bulk_box_path.is_file():
        write_workflow_log(
            "stage_start",
            "bulk",
            current_uuid=current_uuid,
            dump_frequency=dump_frequency_bulk,
            sweep_total=sweep_total_bulk,
            output=sample_bulk_path.name,
        )
        bulk_vf = hsmc.analysis.get_bulk_vf(
            slit_frames,
            box,
            jump=1,
            npoints=50,
            save=figure_path("state_point", ".pdf", current_uuid),
        )

        box_bulk = [(np.pi * n_particle / 6 / vf_init) ** (1.0 / 3.0)] * 3
        system = hsmc.chard_sphere.HSMC(
            n_particle,
            box_bulk,
            [True, True, True],
            [False, False, False],
            r_skin,
        )
        system.fill_hs()
        system.crush(bulk_vf, 0.01)

        print(system)
        print(f"Do particles overlap in Bulk? {system.report_overlap()}")

        for _ in range(sweep_equilibrium):
            system.sweep()

        sample_bulk_path.write_text("")
        with open(sample_bulk_path, "a") as f_xyz:
            for frame in range(sweep_total_bulk):
                system.sweep()
                if frame % dump_frequency_bulk == 0:
                    tmp = system.copy_positions().T
                    np.savetxt(
                        f_xyz,
                        tmp,
                        delimiter=" ",
                        fmt=["A %.8e"] + ["%.8e" for _ in range(2)],
                        header="%s\nframe %s" % (n_particle, frame),
                        comments="",
                    )

        with open(current_bulk_box_path, "w") as f:
            json.dump(system.get_box(), f)

        write_workflow_log(
            "stage_end",
            "bulk",
            current_uuid=current_uuid,
            output=sample_bulk_path.name,
        )
    else:
        write_workflow_log(
            "reuse_existing_output",
            "bulk",
            current_uuid=current_uuid,
            output=sample_bulk_path.name,
        )

    bulk_summary_path = tcc_bulk_path(current_uuid)
    if not bulk_summary_path.is_file():
        tcc_parameters = {
            "voronoi_parameter": 0.82,
            "rcutAA": 1.8,
            "PBCs": 1,
            "Raw": False,
            "clusts": False,
        }
        with open(current_bulk_box_path, "r") as f:
            box_bulk = json.load(f)

        frames_bulk = hsmc.analysis.XYZ(
            sample_bulk_path,
            delimiter=" ",
            usecols=(1, 2, 3),
            engine="pandas",
            align_opt=True,
        )
        tcc_bulk = tcc.OTF()
        tcc_bulk(frames_bulk, box_bulk, **tcc_parameters)

        summary = cluster_population_mapping(tcc_bulk.population.mean(axis=0))
        cluster_names = np.array(list(summary.keys()))
        populations = np.array([summary[name] for name in cluster_names], dtype=float)
        np.savez(bulk_summary_path, cluster_names=cluster_names, populations=populations)


if __name__ == "__main__":
    main()
