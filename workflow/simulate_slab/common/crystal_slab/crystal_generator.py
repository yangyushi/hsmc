"""
crystal_generator.py

Generate bulk crystals from lattice vectors and basis atoms.

Current crystals:
    - FCC
    - HCP

Author: Lars + ChatGPT
"""

import argparse
import numpy as np
from dataclasses import dataclass
from . import geometry as geom

from pathlib import Path

# ============================================================
# Output directory
# ============================================================

SCRIPT_DIR = Path(__file__).parent

OUTPUT_DIR = SCRIPT_DIR / "outputs"

# Defer directory creation until write_xyz is actually called, so importing
# this module does not leave an empty outputs/ directory behind.
# OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# Crystal container
# ============================================================

@dataclass
class Crystal:
    """
    Crystal definition and current configuration.
    """

    # Original unit-cell lattice (never changes)
    unitcell: np.ndarray

    # Basis atoms in fractional coordinates
    basis: np.ndarray

    # Current periodic simulation box
    supercell: np.ndarray = None

    # Cartesian particle positions
    positions: np.ndarray = None

    def make_supercell(self, nx, ny, nz):

        positions = []

        for i in range(nx):
            for j in range(ny):
                for k in range(nz):

                    shift = np.array([i, j, k], dtype=float)

                    for atom in self.basis:

                        frac = atom + shift
                        cart = frac @ self.unitcell

                        positions.append(cart)

        self.positions = np.array(positions)

        self.supercell = np.array([
            self.unitcell[0] * nx,
            self.unitcell[1] * ny,
            self.unitcell[2] * nz,
        ])

        return self


# ============================================================
# Crystal definitions
# ============================================================

def fcc(a):
    """
    Conventional FCC unit cell.
    """

    lattice = np.array([
        [a,0,0],
        [0,a,0],
        [0,0,a]
    ], dtype=float)

    basis = np.array([
        [0,0,0],
        [0,0.5,0.5],
        [0.5,0,0.5],
        [0.5,0.5,0]
    ], dtype=float)

    return Crystal(
        unitcell=lattice,
        basis=basis,
        supercell=lattice.copy(),
    )


def hcp(a, c):
    """
    Conventional HCP unit cell.

    Fractional coordinates correspond to the
    primitive hexagonal cell (2 atoms).
    """

    lattice = np.array([
        [a,              0.0,              0.0],
        [0.5*a,  np.sqrt(3)/2*a,          0.0],
        [0.0,            0.0,               c]
    ])

    # Basis chosen for the 60-degree hexagonal lattice above.  The standard
    # HCP motif (0,0,0) and (2/3,1/3,1/2) is written for the 120-degree
    # convention a2 = (-a/2, a*sqrt(3)/2, 0); with a2 = (+a/2, ...), the
    # equivalent close-packed motif is (1/3, 1/3, 1/2).
    basis = np.array([
        [0.0,      0.0,      0.0],
        [1/3,      1/3,      0.5]
    ])

    return Crystal(
        unitcell=lattice,
        basis=basis,
        supercell=lattice.copy(),
    )


# ============================================================
# Packing fractions
# ============================================================

def fcc_lattice_constant(phi):
    """
    Sphere diameter sigma = 1.
    """

    return (2*np.pi/(3*phi))**(1/3)


def hcp_lattice_constants(phi):
    """
    Ideal HCP (c/a = sqrt(8/3))
    Sphere diameter sigma = 1.
    """

    ratio = np.sqrt(8/3)

    a = (2*np.pi/(3*np.sqrt(3)*ratio*phi))**(1/3)

    c = ratio*a

    return a, c

# ============================================================
# Slab builder
# ============================================================

def build_slab(
    crystal,
    hkl,
    width=None,
    thickness=None,
    n1=None,
    n2=None,
    n3=None,
    vacuum=0.0,
    search=6,
    angle_tol=1.0,
    tol=1e-8,
    rectangular=True,
):
    """
    Build a finite, oriented slab of `crystal` exposing the Miller
    plane (hkl), with independently tunable lateral size, thickness,
    and vacuum gap.

    This is the high-level entry point tying together the geometry
    routines: geom.make_surface_cell finds a (near-)rectangular
    in-plane cell (v1, v2) and minimal out-of-plane repeat v3;
    geom.build_surface_unit_cell fills that cell with atoms;
    geom.assemble_slab tiles it n1 x n2 x n3 times, rotates the whole
    block so (hkl) is horizontal (normal along +z, v1 along +x, v2
    along +y), wraps atoms into the resulting orthogonal box, and
    adds the requested vacuum above the slab.

    Slab size can be specified either as a physical MINIMUM size
    (width/thickness, in the same length units as the lattice
    constant -- rounded UP to the smallest whole number of repeats
    that reaches at least that size), or as an exact number of
    repeats (n1/n2/n3). Explicit n1/n2/n3 take precedence over
    width/thickness if both are given.

    Parameters
    ----------
    crystal : Crystal
        Original bulk crystal (its unitcell/basis are used; its
        current positions/supercell, if any, are ignored).

    hkl : tuple(int, int, int)
        Miller indices of the exposed surface.

    width : float, optional
        Minimum lateral size of the slab, applied to BOTH in-plane
        directions (v1 and v2). Ignored for a given direction if the
        matching n1/n2 is given explicitly.

    thickness : float, optional
        Minimum slab thickness. Ignored if n3 is given explicitly.
        Note one v3 repeat may already bundle several atomic planes
        (see geom.assemble_slab), so the true thickness can exceed
        the requested minimum by up to one repeat's worth.

    n1, n2, n3 : int, optional
        Explicit repeat counts along v1, v2, v3. Override
        width/thickness when given.

    vacuum : float
        Vacuum gap added above the slab along z.

    search, angle_tol, tol, rectangular
        Passed through to geom.make_surface_cell.

    Returns
    -------
    Crystal
        New Crystal with `.positions` (Cartesian atoms) and
        `.supercell` (orthogonal box vectors) describing the slab.
        `.unitcell` and `.basis` are carried over unchanged from the
        input bulk crystal.
    """

    surface = geom.make_surface_cell(
        crystal,
        hkl,
        search=search,
        angle_tol=angle_tol,
        tol=tol,
        rectangular=rectangular,
    )

    v1, v2, v3 = surface.v1, surface.v2, surface.v3
    normal = surface.basis.normal

    cell, unit_positions = geom.build_surface_unit_cell(crystal, surface)

    # --- Resolve repeat counts from physical minimum sizes ---
    if n1 is None:
        if width is None:
            raise ValueError("Must specify either n1 or width.")
        n1 = max(1, int(np.ceil(width / np.linalg.norm(v1) - 1e-9)))

    if n2 is None:
        if width is None:
            raise ValueError("Must specify either n2 or width.")
        n2 = max(1, int(np.ceil(width / np.linalg.norm(v2) - 1e-9)))

    if n3 is None:
        if thickness is None:
            raise ValueError("Must specify either n3 or thickness.")

        layer_spacing = np.dot(v3, normal)  # height added PER repeat

        # A single v3 repeat's atoms don't necessarily span the full
        # layer_spacing themselves (e.g. for FCC(111) they all sit at
        # the same height -- the layer offset is entirely baked into
        # v3's tilt). So n repeats span (n-1)*layer_spacing PLUS
        # whatever a single repeat's atoms already span internally,
        # not n*layer_spacing -- using the latter would silently
        # undershoot the requested thickness by up to one layer.
        proj = unit_positions @ normal
        local_span = proj.max() - proj.min()

        remaining = thickness - local_span

        if remaining <= 0:
            n3 = 1
        else:
            n3 = 1 + int(np.ceil(remaining / layer_spacing - 1e-9))

    supercell, positions = geom.assemble_slab(
        cell,
        unit_positions,
        normal,
        n1,
        n2,
        n3,
        vacuum=vacuum,
    )

    return Crystal(
        unitcell=crystal.unitcell,
        basis=crystal.basis,
        supercell=supercell,
        positions=positions,
    )


# ============================================================
# XYZ writer
# ============================================================

def write_xyz(filename, crystal):

    OUTPUT_DIR.mkdir(exist_ok=True)

    filepath = OUTPUT_DIR / filename

    box = crystal.supercell

    # Extended-XYZ style comment line: embeds the box vectors so the
    # file is self-describing (readable directly by OVITO/VMD/ASE with
    # the correct periodic box, and parseable for verification without
    # needing to remember the original CLI parameters).
    lattice_str = " ".join(
        f"{box[i][j]:.8f}" for i in range(3) for j in range(3)
    )

    comment = (
        f'Lattice="{lattice_str}" '
        f'Properties=species:S:1:pos:R:3 '
        f'pbc="T T F"'
    )

    with open(filepath, "w") as f:

        f.write(f"{len(crystal.positions)}\n")
        f.write(comment + "\n")

        for p in crystal.positions:

            f.write(
                f"A {p[0]:15.8f} {p[1]:15.8f} {p[2]:15.8f}\n"
            )
    
    print(f"Saved: {filepath}")


# ============================================================
# Command-line interface
# ============================================================

def build_argparser():

    parser = argparse.ArgumentParser(
        description=(
            "Generate an oriented, finite crystal slab (FCC or HCP) "
            "for slab simulations, exposing a chosen Miller plane."
        )
    )

    parser.add_argument(
        "--crystal", choices=["fcc", "hcp"], default="fcc",
        help="Bulk crystal structure (default: fcc)",
    )

    parser.add_argument(
        "--phi", type=float, default=0.64,
        help=(
            "Sphere packing fraction, sphere diameter sigma=1 "
            "(default: 0.64). Sets the lattice constant(s)."
        ),
    )

    parser.add_argument(
        "--hkl", type=int, nargs=3, default=(1, 1, 1),
        metavar=("H", "K", "L"),
        help="Miller indices of the exposed surface (default: 1 1 1)",
    )

    parser.add_argument(
        "--width", type=float, default=10.0,
        help=(
            "Minimum lateral slab size, applied to both in-plane "
            "directions, in sigma units (default: 10.0). Rounded up "
            "to the smallest whole number of unit-cell repeats that "
            "reaches at least this size."
        ),
    )

    parser.add_argument(
        "--thickness", type=float, default=10.0,
        help=(
            "Minimum slab thickness, in sigma units (default: 10.0). "
            "Rounded up the same way as --width; the true thickness "
            "can exceed this by up to one out-of-plane repeat."
        ),
    )

    parser.add_argument(
        "--vacuum", type=float, default=15.0,
        help="Vacuum gap added above the slab, sigma units (default: 15.0)",
    )

    parser.add_argument(
        "--oblique", action="store_true",
        help=(
            "Use the minimal (possibly non-rectangular) in-plane cell "
            "instead of searching for a rectangular one. Rarely "
            "needed; mainly useful if the rectangular search fails to "
            "find a good cell for an awkward high-index plane."
        ),
    )

    parser.add_argument(
        "--search", type=int, default=6,
        help=(
            "Brute-force search range for lattice vector combinations "
            "(advanced; default: 6). Increase for high-index (hkl) "
            "planes if a RuntimeError is raised about not finding "
            "enough vectors."
        ),
    )

    parser.add_argument(
        "--angle-tol", type=float, default=1.0,
        help=(
            "Acceptable deviation from 90 degrees when searching for "
            "a rectangular in-plane cell, in degrees (advanced; "
            "default: 1.0)"
        ),
    )

    parser.add_argument(
        "--output", type=str, default=None,
        help=(
            "Output .xyz filename, written under outputs/ (default: "
            "<crystal>_slab_<hkl>.xyz)"
        ),
    )

    return parser


def main():

    parser = build_argparser()
    args = parser.parse_args()

    hkl = tuple(args.hkl)

    if hkl == (0, 0, 0):
        parser.error("--hkl cannot be (0, 0, 0).")

    if args.crystal == "fcc":
        a = fcc_lattice_constant(args.phi)
        bulk = fcc(a)
    else:
        a, c = hcp_lattice_constants(args.phi)
        bulk = hcp(a, c)

    slab = build_slab(
        bulk,
        hkl,
        width=args.width,
        thickness=args.thickness,
        vacuum=args.vacuum,
        search=args.search,
        angle_tol=args.angle_tol,
        rectangular=not args.oblique,
    )

    print(f"Crystal     : {args.crystal.upper()}  (phi = {args.phi})")
    print(f"Miller hkl  : {hkl}")
    print(f"Atoms       : {len(slab.positions)}")
    print("Box vectors :")
    print(slab.supercell)

    if args.output is None:
        hkl_str = "".join(str(abs(x)) for x in hkl)
        filename = f"{args.crystal}_slab_{hkl_str}.xyz"
    else:
        filename = args.output

    write_xyz(filename, slab)


if __name__ == "__main__":
    main()
