#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

import hsmc
from common.slit_setup import create_slit_system
from common.workflow_support import (
    ensure_workflow_metadata,
    figure_path,
    isf_arrays_path,
    isf_metadata_path,
    load_config,
    snap_dump_frequency,
    write_workflow_log,
)


def main():
    conf = load_config()
    metadata = ensure_workflow_metadata()
    current_uuid = metadata["workflow_uuid"]

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
    z_final = float(conf["Boundary"]["z"])
    kind = conf["Boundary"]["kind"]

    write_workflow_log(
        "stage_start",
        "isf",
        current_uuid=current_uuid,
        kind=kind,
        length=length,
        jump=jump,
    )

    state = create_slit_system(
        n_particles,
        sigma,
        vf_init,
        vf_final,
        r_skin,
        vf_crystal,
        z_final,
        kind,
    )
    system = state["system"]
    indices_to_move = state["indices_to_move"]
    box = state["box"]

    print(system)
    print(f"Do particles overlap? {system.report_overlap()}")

    trajectory = np.empty((length, len(indices_to_move), 3))
    for i in range(length):
        for _ in range(jump):
            system.sweep()
        trajectory[i] = system.get_positions().copy().T[indices_to_move]

    time = np.arange(length)
    isf = hsmc.analysis.get_isf_3d(
        trajectory, pbc_box=(box[0], box[1], None), length=length
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
        "kind": kind,
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

    print(f"Relaxation Time: {tau_sweeps:.4f} sweeps")
    print(f"Recommended dump frequency: {recommended_dump_frequency}")

    write_workflow_log(
        "isf_fit",
        "isf",
        current_uuid=current_uuid,
        tau_sweeps=f"{tau_sweeps:.6f}",
        recommended_dump_frequency=recommended_dump_frequency,
        metadata_file=isf_metadata_path(current_uuid).name,
        array_file=isf_arrays_path(current_uuid).name,
    )

    if plot_isf:
        plt.scatter(time, isf, marker="o", color="tomato", fc="none", label="data")
        plt.plot(time, fitted_curve, color="teal", label="fit")
        plt.text(time[len(time) // 2], 0.7, "$\\tau=$" + f"{tau_sweeps:.0f} sweeps")
        plt.xlabel(f"Lag Time / {jump} sweeps")
        plt.ylabel("ISF")
        plt.ylim(-0.1, 1.1)
        plt.tight_layout()
        plt.savefig(figure_path("isf", ".pdf", current_uuid))
        if show_isf:
            plt.show()
        plt.close()

    write_workflow_log(
        "stage_end",
        "isf",
        current_uuid=current_uuid,
        tau_sweeps=f"{tau_sweeps:.6f}",
        recommended_dump_frequency=recommended_dump_frequency,
    )


if __name__ == "__main__":
    main()
