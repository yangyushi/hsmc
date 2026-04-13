#!/usr/bin/env python3
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common.slit_setup import create_slit_system
from common.workflow_support import (
    box_path,
    figure_path,
    load_config,
    resolve_dump_frequency,
    slit_sample_path,
    workflow_uuid,
    write_workflow_log,
)


def main():
    conf = load_config()
    current_uuid = workflow_uuid()

    n_particles = int(conf["System"]["n"])
    sigma = 1
    vf_init = float(conf["System"]["vf_init"])
    vf_final = float(conf["System"]["vf_final"])
    r_skin = float(conf["System"]["r_skin"])
    vf_crystal = float(conf["Boundary"]["vf_crystal"])
    z_final = float(conf["Boundary"]["z"])
    kind = conf["Boundary"]["kind"]
    sweep_equilibrium = int(float(conf["Run"]["equilibrium"]))
    sweep_total = int(float(conf["Run"]["total"]))
    dump_frequency_config = conf["Run"]["dump_frequency"]
    sample_path = slit_sample_path(conf, current_uuid)
    slit_box_path = box_path(current_uuid)

    if sample_path.is_file() and slit_box_path.is_file():
        write_workflow_log(
            "reuse_existing_output",
            "simulate",
            current_uuid=current_uuid,
            output=sample_path.name,
        )
        return

    dump_frequency = resolve_dump_frequency(dump_frequency_config, current_uuid)
    write_workflow_log(
        "resolved_dump_frequency",
        "simulate",
        current_uuid=current_uuid,
        configured_value=dump_frequency_config,
        dump_frequency=dump_frequency,
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
    walls = state["walls"]
    gas = state["gas"]

    if walls is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*state["configuration"].T)
    else:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(*walls.T)
        ax.scatter(*gas.T)
    plt.tight_layout()
    plt.savefig(figure_path("system_start", ".png", current_uuid))
    plt.close(fig)

    print(system)
    print(f"Do particles overlap? {system.report_overlap()}")

    pos = system.copy_positions()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    ax.scatter(*pos, color="w", ec="k", marker="o")
    plt.title(rf"{kind}, $\phi$ = {vf_final}")
    ax.set_xlabel(r"X / $\sigma$")
    ax.set_ylabel(r"Y / $\sigma$")
    ax.set_zlabel(r"Z / $\sigma$")
    ax2.hist(pos[2], bins=250)
    ax2.set_xlabel(r"Z / $\sigma$")
    ax2.set_ylabel("PDF")
    plt.tight_layout()
    plt.savefig(figure_path("system_crushed", ".png", current_uuid))
    plt.close(fig)

    with open(slit_box_path, "w") as f:
        json.dump(system.get_box(), f)

    print("reaching equilibrium")
    for _ in tqdm(range(sweep_equilibrium)):
        system.sweep()

    sample_path.write_text("")
    with open(sample_path, "a") as f_xyz:
        print("starting sampling")
        write_workflow_log(
            "stage_start",
            "simulate",
            current_uuid=current_uuid,
            dump_frequency=dump_frequency,
            sweep_total=sweep_total,
            output=sample_path.name,
        )
        n_move = len(indices_to_move)
        for frame in tqdm(range(sweep_total)):
            system.sweep()
            if frame % dump_frequency == 0:
                tmp = system.copy_positions().T[indices_to_move]
                np.savetxt(
                    f_xyz,
                    tmp,
                    delimiter=" ",
                    fmt=["A %.8e"] + ["%.8e" for _ in range(2)],
                    header="%s\nframe %s" % (n_move, frame),
                    comments="",
                )

    write_workflow_log(
        "stage_end",
        "simulate",
        current_uuid=current_uuid,
        dump_frequency=dump_frequency,
        output=sample_path.name,
    )


if __name__ == "__main__":
    main()
