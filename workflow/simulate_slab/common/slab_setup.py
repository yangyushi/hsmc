"""Helpers for constructing the central-slab simulation configuration.

The workflow builds a fully periodic box containing a fixed crystal slab
surrounded by bulk fluid.  The target state point ``vf_final`` is the fluid
volume fraction measured in the bulk region, i.e. the fluid far away from the
crystal surfaces.  A ``boundary_thickness`` next to each surface is therefore
excluded from the volume used to size the box.

hsmc is still constructed with ``scale_fixed_particles=True`` and the isotropic
scaling workaround below is used to prepare a dense fluid around the crystal
without getting stuck in MC overlap removal.
"""

from __future__ import annotations

import math
from typing import Protocol, TypedDict

import numpy as np

import hsmc
from common.crystal_slab.slab_search import find_slab_geometry


class SlabSimulationSystem(Protocol):
    """Protocol for the simulation system used by the slab workflow."""

    def load_positions(self, positions: np.ndarray) -> None:
        """Load particle positions into the simulation system."""

    def set_indices(self, indices: np.ndarray) -> None:
        """Restrict mobile particles to the given indices."""

    def fill_hs(self) -> None:
        """Initialize hard-sphere data structures."""

    def crush(
        self,
        volume_fraction: float,
        rate: float,
    ) -> None:
        """Isotropically compress the system."""

    def sweep(self) -> None:
        """Perform one Monte Carlo sweep."""

    def report_overlap(self) -> bool:
        """Report whether particles overlap."""

    def copy_positions(self) -> np.ndarray:
        """Return a copy of particle positions."""

    def get_box(self) -> list[float]:
        """Return the current box dimensions."""


class SlabSystemSetup(TypedDict):
    """Initial system state returned to workflow stages before simulation starts."""

    system: SlabSimulationSystem
    box: np.ndarray
    indices_to_move: np.ndarray
    configuration: np.ndarray
    slab: np.ndarray
    fluid: np.ndarray | None
    slab_info: dict


def _place_fluid(
    n_fluid: int,
    box: np.ndarray,
    z_slab: float,
    scale: float,
    sigma: float = 1.0,
    buffer: float = 1.0,
) -> np.ndarray:
    """Place fluid particles uniformly in the free volume above and below the slab.

    The slab is centered in the box (z in [z_c - z_slab/2, z_c + z_slab/2]).
    Because the initial box is ``scale`` times larger than the final box, we
    seed fluid outside the *scaled* slab so that after isotropic compression it
    ends up outside the final slab.
    """
    Lx, Ly, Lz = box
    z_c = Lz / 2.0
    half_slab_init = z_slab / (2.0 * scale)
    z_lower_min = z_c - half_slab_init - buffer
    z_lower_max = z_c - half_slab_init + buffer
    z_upper_min = z_c + half_slab_init - buffer
    z_upper_max = z_c + half_slab_init + buffer

    # Total free height in the initial box excluding the scaled slab and buffers.
    free_height = max(0.0, z_lower_min) + max(0.0, Lz - z_upper_max)
    if free_height <= 0:
        raise ValueError(
            f"No room for fluid: Lz={Lz:.3f}, z_slab={z_slab:.3f}, "
            f"scale={scale:.3f}, buffer={buffer:.3f}"
        )

    # Place particles uniformly in the free volume.
    positions = np.random.uniform(0.0, 1.0, size=(n_fluid, 3))
    positions[:, 0] *= Lx
    positions[:, 1] *= Ly

    # Split the random z coordinate proportionally between lower and upper regions.
    lower_fraction = max(0.0, z_lower_min) / free_height
    z_rand = positions[:, 2]
    in_lower = z_rand < lower_fraction
    positions[in_lower, 2] = z_rand[in_lower] * free_height
    positions[~in_lower, 2] = z_upper_max + (z_rand[~in_lower] - lower_fraction) * free_height

    return positions


def create_slab_system(
    n_particles: int,
    sigma: float,
    vf_init: float,
    vf_final: float,
    r_skin: float,
    vf_crystal: float,
    slab_thickness: float,
    z_aspect: float,
    min_gap: float,
    boundary_thickness: float,
    crystal: str,
    hkl: tuple[int, int, int],
) -> SlabSystemSetup:
    """Build the central-slab system and return all state needed by later stages."""

    candidate = find_slab_geometry(
        n_fluid=n_particles,
        vf_final=vf_final,
        crystal_name=crystal,
        hkl=hkl,
        vf_crystal=vf_crystal,
        slab_thickness=slab_thickness,
        z_aspect=z_aspect,
        min_gap=min_gap,
        boundary_thickness=boundary_thickness,
        sigma=sigma,
    )

    Lx = candidate.Lx
    Ly = candidate.Ly
    z_final = candidate.z_final
    z_slab = candidate.z_slab
    slab = candidate.crystal.positions.copy()
    N_crystal = candidate.N_crystal

    V_particle = math.pi * sigma**3 / 6.0
    V_fluid = n_particles * V_particle
    V_crystal = N_crystal * V_particle
    V_total = V_fluid + V_crystal
    vf_total_final = V_total / (Lx * Ly * z_final)

    # Compute the isotropic scale factor that brings the system from the
    if vf_total_final <= vf_init:
        raise ValueError(
            f"Final total volume fraction {vf_total_final:.4f} is not greater "
            f"than initial volume fraction {vf_init:.4f}. Increase vf_final, "
            "slab_thickness, or n_particles."
        )
    scale = math.pow(vf_total_final / vf_init, -1.0 / 3.0)
    if scale >= 1.0:
        raise ValueError(f"Invalid compression scale {scale}; check vf_init.")

    box_init = np.array([Lx / scale, Ly / scale, z_final / scale])
    z_c_init = box_init[2] / 2.0
    slab_init = slab / scale
    slab_init[:, 2] += z_c_init - (slab_init[:, 2].max() + slab_init[:, 2].min()) / 2.0

    fluid = _place_fluid(
        n_particles,
        box_init,
        z_slab,
        scale,
        sigma=sigma,
        buffer=sigma,
    )

    total_particles = n_particles + N_crystal
    configuration = np.concatenate((slab_init, fluid), axis=0)
    indices_to_move = np.arange(N_crystal, total_particles)

    is_pbc = [True, True, True]
    is_hard = [False, False, False]
    system = hsmc.chard_sphere.HSMC(
        total_particles, box_init, is_pbc, is_hard, r_skin,
        scale_fixed_particles=True,
    )
    system.load_positions(configuration.T)
    system.set_indices(indices_to_move)
    system.fill_hs()

    system.crush(vf_total_final, 0.01)

    final_box = np.array(system.get_box())
    if not all(
        math.isclose(final_box[i], expected, rel_tol=1e-6)
        for i, expected in enumerate([Lx, Ly, z_final])
    ):
        raise RuntimeError(
            f"Unexpected final box after crush: {final_box.tolist()} vs "
            f"[{Lx}, {Ly}, {z_final}]"
        )

    positions = system.copy_positions()
    slab_z_mid = (positions[2, :N_crystal].max() + positions[2, :N_crystal].min()) / 2.0
    shift_z = final_box[2] / 2.0 - slab_z_mid
    positions[2, :] += shift_z
    positions[2, :] %= final_box[2]
    system.load_positions(positions)
    system.set_indices(indices_to_move)

    if system.report_overlap():
        raise RuntimeError(
            "Overlap remained after isotropic crush; the initial configuration "
            "could not be equilibrated."
        )

    slab_info = {
        "Lx": Lx,
        "Ly": Ly,
        "z_final": z_final,
        "z_slab": z_slab,
        "N_crystal": N_crystal,
        "vf_total_final": vf_total_final,
        "scale": scale,
        "box_init": box_init.tolist(),
        "n1": candidate.n1,
        "n2": candidate.n2,
        "n3": candidate.n3,
        "aspect_ratio": candidate.aspect_ratio,
        "z_aspect_actual": candidate.z_aspect_actual,
        "boundary_thickness": boundary_thickness,
        "gap": candidate.gap,
    }

    return {
        "system": system,
        "box": final_box,
        "indices_to_move": indices_to_move,
        "configuration": system.copy_positions().T,
        "slab": system.copy_positions().T[:N_crystal],
        "fluid": fluid,
        "slab_info": slab_info,
    }
