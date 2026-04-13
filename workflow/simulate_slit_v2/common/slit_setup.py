#!/usr/bin/env python3
"""Helpers for constructing the initial slit simulation configuration."""

from typing import Literal, Protocol, TypedDict

import numpy as np

import hsmc


VALID_PLANES = ("fcc100", "fcc111", "fcc110", "hcp110")
BoundaryKind = Literal["hardwall", "fcc100", "fcc111", "fcc110", "hcp110"]


class SlitSimulationSystem(Protocol):
    """Protocol for the simulation system used by the slit workflow."""

    def load_positions(self, positions: np.ndarray) -> None:
        """Load particle positions into the simulation system."""

    def set_indices(self, indices: np.ndarray) -> None:
        """Restrict mobile particles to the given indices."""

    def fill_hs(self) -> None:
        """Initialize hard-sphere data structures."""

    def crush_along_axis(
        self,
        volume_fraction: float,
        rate: float,
        axis: int,
    ) -> None:
        """Compress the system along one axis."""

    def sweep(self) -> None:
        """Perform one Monte Carlo sweep."""

    def report_overlap(self) -> bool:
        """Report whether particles overlap."""

    def copy_positions(self) -> np.ndarray:
        """Return a copy of particle positions."""

    def get_box(self) -> list[float]:
        """Return the current box dimensions."""


class SlitSystemSetup(TypedDict):
    """Initial system state returned to workflow stages before simulation starts."""

    system: SlitSimulationSystem
    box: np.ndarray
    indices_to_move: np.ndarray
    configuration: np.ndarray
    walls: np.ndarray | None
    gas: np.ndarray | None


def create_slit_system(
    n_particles: int,
    sigma: float,
    vf_init: float,
    vf_final: float,
    r_skin: float,
    vf_crystal: float,
    z_final: float,
    kind: BoundaryKind,
) -> SlitSystemSetup:
    """Build the slit system and return all state needed by later workflow stages."""

    v_sph = np.pi * n_particles * sigma ** 3 / 6
    l_xy = np.sqrt(v_sph / vf_final / z_final)

    walls = None
    gas = None

    if kind == "hardwall":
        z_init = v_sph / vf_init / l_xy ** 2
        box = np.array((l_xy, l_xy, z_init))
        configuration = np.random.uniform(0, 1, (n_particles, 3)) * box
        indices_to_move = np.arange(n_particles)
    elif kind in VALID_PLANES:
        crystal_kind = hsmc.crystal.parse_plane_kind(kind)
        lattice_constant = hsmc.crystal.get_lattice_constant(
            crystal_kind, vf=vf_crystal, sigma=sigma
        )
        uc = np.array(hsmc.crystal.plane_info[kind]["unit_cell"][:2])
        nx = int(np.floor(l_xy / (uc[0] * lattice_constant)))
        ny = int(np.ceil(l_xy / (uc[1] * lattice_constant)))

        plane, box_xy = hsmc.crystal.get_plane(
            kind, nx, ny, vf=vf_crystal, sigma=sigma
        )
        plane = np.concatenate((plane, np.zeros((len(plane), 1))), axis=1)

        v_sph = np.pi * (n_particles + len(plane) * 2) * sigma ** 3 / 6
        a_xy = box_xy[0] * box_xy[1]
        box_z = v_sph / a_xy / vf_init
        walls = np.concatenate((plane, plane + np.array((0, 0, box_z))), axis=0)
        box = np.array((box_xy[0], box_xy[1], box_z))

        gas = np.random.uniform(0, 1, size=(n_particles, 3))
        gas *= box - np.array((0, 0, sigma * 2))
        gas = gas + np.array((0, 0, sigma))

        total_particles = n_particles + len(walls)
        configuration = np.concatenate((walls, gas))
        indices_to_move = np.arange(len(walls), total_particles)
        n_particles = total_particles
    else:
        raise ValueError(f"invalid boundary kind: {kind}")

    is_pbc = [True, True, False]
    is_hard = [False, False, True]
    system = hsmc.chard_sphere.HSMC(n_particles, box, is_pbc, is_hard, r_skin)
    system.load_positions(configuration.T)
    system.set_indices(indices_to_move)
    system.fill_hs()
    system.crush_along_axis(vf_final, 0.01, 2)

    return {
        "system": system,
        "box": box,
        "indices_to_move": indices_to_move,
        "configuration": configuration,
        "walls": walls,
        "gas": gas,
    }
