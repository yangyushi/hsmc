#!/usr/bin/env python3
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from common.slab_setup import create_slab_system
from common.workflow_support import (
    active_workflow_uuid,
    box_path,
    figure_path,
    get_workflow_logger,
    load_config,
    resolve_dump_frequency,
    slab_info_path,
    slit_sample_path,
)


def _visualise_with_ghosts(positions: np.ndarray, box: np.ndarray, pad: float = 2.0) -> np.ndarray:
    """Return positions plus periodic ghost copies near each box boundary.

    This is only for visualisation; it makes finite crystals look like they
    span the full periodic box by drawing atoms that wrap across boundaries.
    """
    Lx, Ly, Lz = box
    ghosts = [positions.copy()]
    for dx in (-Lx, 0.0, Lx):
        for dy in (-Ly, 0.0, Ly):
            if dx == 0.0 and dy == 0.0:
                continue
            shifted = positions + [dx, dy, 0.0]
            # Keep only atoms that fall within a padding distance of the
            # opposite boundary so the plot does not blow up.
            keep = (
                (shifted[:, 0] < pad)
                | (shifted[:, 0] > Lx - pad)
                | (shifted[:, 1] < pad)
                | (shifted[:, 1] > Ly - pad)
            )
            ghosts.append(shifted[keep])
    return np.concatenate(ghosts, axis=0)


def main():
    conf = load_config()
    current_uuid = active_workflow_uuid()
    logger = get_workflow_logger("simulate", current_uuid)

    n_particles = int(conf["System"]["n"])
    sigma = 1
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
    sweep_equilibrium = int(float(conf["Run"]["equilibrium"]))
    sweep_total = int(float(conf["Run"]["total"]))
    dump_frequency_config = conf["Run"]["dump_frequency"]
    sample_path = slit_sample_path(conf, current_uuid)
    slab_box_path = box_path(current_uuid)

    if sample_path.is_file() and slab_box_path.is_file():
        logger.info("Reusing existing output: %s", sample_path.name)
        return

    dump_frequency = resolve_dump_frequency(dump_frequency_config, current_uuid)
    logger.info(
        "Resolved dump frequency: %s -> %s",
        dump_frequency_config,
        dump_frequency,
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
    slab = state["slab"]
    fluid = state["fluid"]
    box = state["box"]
    Lx, Ly, z_final = box

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(121, projection="3d")
    slab_plot = _visualise_with_ghosts(slab, box)
    ax.scatter(*slab_plot.T, s=10, label="crystal")
    if fluid is not None:
        ax.scatter(*fluid.T, s=5, label="fluid (initial)")
    ax.set_xlabel(r"X / $\sigma$")
    ax.set_ylabel(r"Y / $\sigma$")
    ax.set_zlabel(r"Z / $\sigma$")
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.legend()

    ax2 = fig.add_subplot(122)
    ax2.hist(state["configuration"][:, 2], bins=250)
    ax2.set_xlabel(r"Z / $\sigma$")
    ax2.set_ylabel("PDF")
    plt.tight_layout()
    plt.savefig(figure_path("system_start", ".png", current_uuid))
    plt.close(fig)

    logger.info("System initialized: %s", system)
    logger.info("Particle overlap detected: %s", system.report_overlap())

    pos = system.copy_positions()
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    ax.scatter(*pos, color="w", ec="k", marker="o", s=5)
    plt.title(rf"{crystal} {hkl}, $\phi$ = {vf_final}")
    ax.set_xlabel(r"X / $\sigma$")
    ax.set_ylabel(r"Y / $\sigma$")
    ax.set_zlabel(r"Z / $\sigma$")
    ax2.hist(pos[2], bins=250)
    ax2.set_xlabel(r"Z / $\sigma$")
    ax2.set_ylabel("PDF")
    plt.tight_layout()
    plt.savefig(figure_path("system_crushed", ".png", current_uuid))
    plt.close(fig)

    with open(slab_box_path, "w") as f:
        json.dump(system.get_box(), f)

    with open(slab_info_path(current_uuid), "w") as f:
        json.dump(state["slab_info"], f, indent=2)

    logger.info("Reaching equilibrium")
    for _ in tqdm(range(sweep_equilibrium), file=sys.stderr):
        system.sweep()

    sample_path.write_text("")
    with open(sample_path, "a") as f_xyz:
        logger.info(
            "Starting sampling: dump_frequency=%s, sweep_total=%s, output=%s",
            dump_frequency,
            sweep_total,
            sample_path.name,
        )
        n_crystal = len(slab)
        for frame in tqdm(range(sweep_total), file=sys.stderr):
            system.sweep()
            if frame % dump_frequency == 0:
                positions = system.copy_positions().T
                crystal_pos = positions[:n_crystal]
                fluid_pos = positions[n_crystal:]
                n_total = len(positions)
                f_xyz.write(f"{n_total}\nframe {frame}\n")
                for p in crystal_pos:
                    f_xyz.write(
                        f"B {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n"
                    )
                for p in fluid_pos:
                    f_xyz.write(
                        f"A {p[0]:.8e} {p[1]:.8e} {p[2]:.8e}\n"
                    )

    logger.info(
        "Completed sampling: dump_frequency=%s, output=%s",
        dump_frequency,
        sample_path.name,
    )


if __name__ == "__main__":
    main()
