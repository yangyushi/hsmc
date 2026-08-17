"""Vendored crystal-slab-generator code with a thin wrapper layer."""

from .crystal_generator import (
    Crystal,
    build_slab,
    fcc,
    fcc_lattice_constant,
    hcp,
    hcp_lattice_constants,
    write_xyz,
)
from .geometry import (
    SurfaceCell,
    build_surface_unit_cell,
    cart_to_frac,
    find_plane_vectors,
    find_rectangular_surface_cell,
    frac_to_cart,
    make_surface_cell,
    miller_normal,
)

__all__ = [
    "Crystal",
    "build_slab",
    "fcc",
    "fcc_lattice_constant",
    "hcp",
    "hcp_lattice_constants",
    "write_xyz",
    "SurfaceCell",
    "build_surface_unit_cell",
    "cart_to_frac",
    "find_plane_vectors",
    "find_rectangular_surface_cell",
    "frac_to_cart",
    "make_surface_cell",
    "miller_normal",
]
