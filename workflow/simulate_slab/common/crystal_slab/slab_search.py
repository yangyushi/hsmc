"""Search for a square-ish, PBC-compatible crystal slab embedded in a fluid box."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .crystal_generator import build_slab, fcc, fcc_lattice_constant, hcp, hcp_lattice_constants
from .geometry import build_surface_unit_cell, make_surface_cell


@dataclass
class SlabCandidate:
    """A candidate slab geometry."""

    crystal: object
    n1: int
    n2: int
    n3: int
    Lx: float
    Ly: float
    z_slab: float
    N_crystal: int
    z_final: float
    aspect_ratio: float
    z_aspect_actual: float
    gap: float
    score: float


def _make_bulk(crystal_name: str, vf_crystal: float, sigma: float = 1.0):
    """Create a conventional bulk crystal at the requested crystal vf."""
    if crystal_name == "fcc":
        a = fcc_lattice_constant(vf_crystal) * sigma
        return fcc(a)
    if crystal_name == "hcp":
        a, c = hcp_lattice_constants(vf_crystal)
        a *= sigma
        c *= sigma
        return hcp(a, c)
    raise ValueError(f"Unsupported crystal: {crystal_name}")


def _compute_n3(crystal, hkl, thickness, search, angle_tol):
    """Determine the number of out-of-plane repeats for the requested thickness."""
    surface = make_surface_cell(
        crystal, hkl, search=search, angle_tol=angle_tol, rectangular=True
    )
    cell, unit_positions = build_surface_unit_cell(crystal, surface)
    v3 = surface.v3
    normal = surface.basis.normal
    layer_spacing = np.dot(v3, normal)
    proj = unit_positions @ normal
    local_span = float(proj.max() - proj.min())
    remaining = thickness - local_span
    if remaining <= 0:
        return 1
    return 1 + int(math.ceil(remaining / layer_spacing - 1e-9))


def _candidate_score(
    Lx: float,
    Ly: float,
    z_final: float,
    z_aspect: float,
) -> float:
    """Score a candidate by square-ness and z-elongation."""
    aspect_ratio = max(Lx / Ly, Ly / Lx)
    return (aspect_ratio - 1.0) ** 2 + (z_final / (z_aspect * Lx) - 1.0) ** 2


def find_slab_geometry(
    n_fluid: int,
    vf_final: float,
    crystal_name: str,
    hkl: tuple[int, int, int],
    vf_crystal: float,
    slab_thickness: float,
    z_aspect: float = 4.0,
    min_gap: float = 5.0,
    boundary_thickness: float = 0.0,
    sigma: float = 1.0,
    search: int = 6,
    angle_tol: float = 1.0,
    window_factor: float = 2.0,
) -> SlabCandidate:
    """Find a slab geometry that is square-ish in XY and elongated in Z.

    A positive ``boundary_thickness`` excludes a layer of that thickness
    next to each crystal surface from the state-point volume.  The target
    fluid volume fraction is then enforced in the remaining bulk region
    far away from the slab.
    """
    if n_fluid <= 0:
        raise ValueError("n_fluid must be positive")
    if not 0 < vf_final < 1:
        raise ValueError("vf_final must be between 0 and 1")
    if slab_thickness <= 0:
        raise ValueError("slab_thickness must be positive")
    if boundary_thickness < 0:
        raise ValueError("boundary_thickness must be non-negative")

    bulk = _make_bulk(crystal_name, vf_crystal, sigma=sigma)
    n3 = _compute_n3(bulk, hkl, slab_thickness, search, angle_tol)

    # Surface unit cell vectors and area (one in-plane repeat).
    surface = make_surface_cell(
        bulk, hkl, search=search, angle_tol=angle_tol, rectangular=True
    )
    v1 = surface.v1
    v2 = surface.v2
    v1_len = float(np.linalg.norm(v1))
    v2_len = float(np.linalg.norm(v2))

    # Effective bulk volume for the state point: free volume minus the
    # boundary layers adjacent to the two crystal surfaces.
    V_particle = math.pi * sigma**3 / 6.0
    V_fluid = n_fluid * V_particle
    A_seed = (V_fluid / vf_final / z_aspect) ** (2.0 / 3.0)

    n1_seed = math.sqrt(A_seed) / v1_len
    n2_seed = math.sqrt(A_seed) / v2_len

    n1_min = max(1, int(n1_seed / window_factor))
    n1_max = max(n1_min + 1, int(n1_seed * window_factor))
    n2_min = max(1, int(n2_seed / window_factor))
    n2_max = max(n2_min + 1, int(n2_seed * window_factor))

    best: SlabCandidate | None = None
    best_score = float("inf")

    for n1 in range(n1_min, n1_max + 1):
        for n2 in range(n2_min, n2_max + 1):
            try:
                slab = build_slab(
                    bulk,
                    hkl,
                    n1=n1,
                    n2=n2,
                    n3=n3,
                    vacuum=0.0,
                    search=search,
                    angle_tol=angle_tol,
                    rectangular=True,
                )
            except Exception:
                continue

            Lx = float(slab.supercell[0, 0])
            Ly = float(slab.supercell[1, 1])
            z_slab = float(slab.supercell[2, 2])
            N_crystal = len(slab.positions)

            V_crystal = N_crystal * V_particle
            # Effective bulk volume for the state point: free volume minus the
            # two boundary layers adjacent to the crystal surfaces.  The boundary
            # layers are not empty; assuming a roughly linear density recovery
            # from the surface to the bulk, their average density is about half
            # the bulk density.  Solving for z_final so that the bulk region
            # (distance > boundary_thickness from a surface) has volume
            # fraction vf_final gives:
            #   z_final = z_slab + boundary_thickness
            #             + V_fluid / (vf_final * Lx * Ly)
            z_final = (
                z_slab
                + boundary_thickness
                + V_fluid / (vf_final * Lx * Ly)
            )

            gap = (z_final - z_slab) / 2.0
            if gap < min_gap + boundary_thickness:
                continue

            score = _candidate_score(Lx, Ly, z_final, z_aspect)

            if score < best_score:
                best_score = score
                best = SlabCandidate(
                    crystal=slab,
                    n1=n1,
                    n2=n2,
                    n3=n3,
                    Lx=Lx,
                    Ly=Ly,
                    z_slab=z_slab,
                    N_crystal=N_crystal,
                    z_final=z_final,
                    aspect_ratio=max(Lx / Ly, Ly / Lx),
                    z_aspect_actual=z_final / Lx,
                    gap=gap,
                    score=score,
                )

    if best is None:
        raise RuntimeError(
            f"Could not find a valid slab geometry for {crystal_name} {hkl} "
            f"with n_fluid={n_fluid}, vf_final={vf_final}, "
            f"slab_thickness={slab_thickness}, z_aspect={z_aspect}, "
            f"min_gap={min_gap}, boundary_thickness={boundary_thickness}. "
            f"Try adjusting n_fluid, slab_thickness, z_aspect, min_gap, or "
            "boundary_thickness."
        )

    return best
