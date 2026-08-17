#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import hsmc
from common.slab_setup import create_slab_system
from common.workflow_support import (
    active_workflow_uuid,
    figure_path,
    get_workflow_logger,
    isf_arrays_path,
    isf_metadata_path,
    load_config,
    snap_dump_frequency,
)


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("isf", current_uuid)

    length = int(conf["ISF"]["length"])
    jump = int(conf["ISF"]["jump"])
    show_isf = conf.getboolean("ISF", "show_isf")
    plot_isf = conf.getboolean("ISF", "plot_isf")

    sigma = 1
    n_particles = int(conf["ISF"]["n"])
    vf_init = float(conf["System"]["vf_init"])
    vf_final = float(conf["System"]["vf_final"])
    r_skin = float(conf["System"]["r_skin"])
    vf_crystal = float(conf["Boundary"]["vf_crystal"])
    slab_thickness = float(conf["Boundary"]["slab_thickness"])
    z_aspect = float(conf["Boundary"]["z_aspect"])
    min_gap = float(conf["Boundary"]["min_gap"])
    boundary_thickness = float(conf["Boundary"]["boundary_thickness"])
    crystal = conf["Boundary"]["crystal"].strip().lower()
    hkl_raw = conf["Boundary"]["hkl"].split()
    hkl = tuple(int(x) for x in hkl_raw)
    logger.info(
        "Building ISF system: %s %s, length=%s, jump=%s",
        crystal,
        hkl,
        length,
        jump,
    )

    state = create_slab_system(
        n_particles,
        sigma,
        vf_init,
        vf_final,
        r_skin,
        vf_crystal,
        slab_thickness,
        z_aspect,
        min_gap,
        boundary_thickness,
        crystal,
        hkl,
    )
    system = state["system"]
    indices_to_move = state["indices_to_move"]
    box = state["box"]

    logger.info("System initialized: %s", system)
    logger.info("Particle overlap detected: %s", system.report_overlap())

    trajectory = np.empty((length, len(indices_to_move), 3))
    for i in range(length):
        for _ in range(jump):
            system.sweep()
        trajectory[i] = system.get_positions().copy().T[indices_to_move]

    time = np.arange(length)
    isf = hsmc.analysis.get_isf_3d(
        trajectory, pbc_box=(box[0], box[1], box[2]), length=length
    )
    tau_frames, stretching_exponent = curve_fit(
        f=lambda x, tau, b: np.exp(-(x / tau) ** b),
        xdata=time,
        ydata=isf,
        p0=(10, 1),
    )[0]
    tau_sweeps = float(tau_frames * jump)
    recommended_dump_frequency = snap_dump_frequency(tau_sweeps)
    fitted_curve = np.exp(-(time / tau_frames) ** stretching_exponent)

    np.savez(
        isf_arrays_path(current_uuid),
        time=time,
        isf=isf,
        fitted_curve=fitted_curve,
    )

    isf_metadata = {
        "workflow_uuid": current_uuid,
        "crystal": crystal,
        "hkl": list(hkl),
        "vf_final": vf_final,
        "isf_particle_count": int(len(indices_to_move)),
        "length": length,
        "jump": jump,
        "tau_frames": float(tau_frames),
        "tau_sweeps": tau_sweeps,
        "stretching_exponent": float(stretching_exponent),
        "fitted": True,
        "recommended_dump_frequency": recommended_dump_frequency,
        "array_file": isf_arrays_path(current_uuid).name,
    }
    with open(isf_metadata_path(current_uuid), "w") as f:
        json.dump(isf_metadata, f, indent=2)

    logger.info("Relaxation time: %.4f sweeps", tau_sweeps)
    logger.info(
        "Recommended dump frequency: %s",
        recommended_dump_frequency,
    )
    logger.info(
        "Saved ISF fit: tau_sweeps=%.6f, dump_frequency=%s, metadata=%s, arrays=%s",
        tau_sweeps,
        recommended_dump_frequency,
        isf_metadata_path(current_uuid).name,
        isf_arrays_path(current_uuid).name,
    )

    if plot_isf:
        plt.scatter(
            time, isf, marker="o",
            color="tomato", fc="none", label="data"
        )
        plt.plot(
            time, fitted_curve, color="teal", label="fit"
        )
        plt.text(
            time[len(time) // 2], 0.7,
            "$\\tau=$" + f"{tau_sweeps:.0f} sweeps"
        )
        plt.xlabel(f"Lag Time / {jump} sweeps")
        plt.ylabel("ISF")
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(figure_path("isf", ".pdf", current_uuid))
        if show_isf:
            plt.show()
        plt.close()
    logger.info("Completed ISF analysis")


if __name__ == "__main__":
    main()
