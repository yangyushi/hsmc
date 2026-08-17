"""
geometry.py

General geometry routines used throughout the crystal package.
"""

import numpy as np
from scipy.spatial.transform import Rotation


from dataclasses import dataclass

@dataclass
class PlaneBasis:
    hkl: tuple[int, int, int]
    normal: np.ndarray

    coeff1: np.ndarray
    coeff2: np.ndarray

    v1: np.ndarray
    v2: np.ndarray

@dataclass
class SurfaceCell:
    """
    Lattice basis adapted to a Miller plane.
    """

    basis: PlaneBasis

    v1: np.ndarray
    v2: np.ndarray
    v3: np.ndarray

    coeff3: np.ndarray

def frac_to_cart(frac, lattice):
    """
    Fractional -> Cartesian coordinates.
    """

    return np.asarray(frac) @ lattice


def cart_to_frac(cart, lattice):
    """
    Cartesian -> Fractional coordinates.
    """

    return np.asarray(cart) @ np.linalg.inv(lattice)


def cell_volume(lattice):
    """
    Unit-cell volume.
    """

    return np.linalg.det(lattice)


def reciprocal_lattice(lattice):
    """
    Reciprocal lattice vectors.

    Rows contain b1, b2, b3.
    """

    return 2*np.pi*np.linalg.inv(lattice).T


def miller_normal(lattice, hkl):
    """
    Unit normal vector of a Miller plane.
    """

    B = reciprocal_lattice(lattice)

    h, k, l = hkl

    n = h*B[0] + k*B[1] + l*B[2]

    return n / np.linalg.norm(n)

def _in_plane_lattice_vectors(lattice, normal, search=4, tol=1e-10):
    """
    Brute-force enumeration of lattice vectors (integer combinations
    of the unit cell, from -search ... +search) that lie within the
    plane defined by `normal`.

    Parameters
    ----------
    lattice : (3,3) ndarray
        Unit-cell lattice vectors (rows).

    normal : (3,) ndarray
        Unit normal vector of the plane.

    search : int
        Search integer combinations from -search ... +search.

    tol : float
        Tolerance for determining whether a vector lies in the plane.

    Returns
    -------
    list of (coeff, vec)
        Candidates sorted by ascending vector length. `coeff` is the
        integer (3,) combination, `vec` is the corresponding Cartesian
        lattice vector.
    """

    candidates = []

    for i in range(-search, search + 1):
        for j in range(-search, search + 1):
            for k in range(-search, search + 1):

                if i == 0 and j == 0 and k == 0:
                    continue

                coeff = np.array([i, j, k])

                vec = coeff @ lattice

                # Keep vectors lying in the plane
                if abs(np.dot(vec, normal)) < tol:

                    candidates.append((coeff, vec))

    candidates.sort(key=lambda c: np.linalg.norm(c[1]))

    return candidates


def find_plane_vectors(crystal, hkl, search=4, tol=1e-10):
    """
    Find the two shortest independent lattice vectors lying
    in the Miller plane (hkl).

    Parameters
    ----------
    crystal : Crystal
        Crystal object containing the unit cell.

    hkl : tuple(int, int, int)
        Miller indices.

    search : int
        Search integer combinations from -search ... +search.

    tol : float
        Tolerance for determining whether a vector lies in the plane.

    Returns
    -------
    PlaneBasis
        Information describing the 2D lattice of the Miller plane.
    """

    lattice = crystal.unitcell

    normal = miller_normal(lattice, hkl)

    candidates = _in_plane_lattice_vectors(lattice, normal, search=search, tol=tol)

    if len(candidates) < 2:
        raise RuntimeError("Could not find enough in-plane vectors.")

    c1, v1 = candidates[0]

    for c2, v2 in candidates[1:]:
        if np.linalg.norm(np.cross(v1, v2)) > tol:
            return PlaneBasis(
                hkl=hkl,
                normal=normal,
                coeff1=c1,
                coeff2=c2,
                v1=v1,
                v2=v2,
            )

    raise RuntimeError("Could not find two independent vectors.")

def find_rectangular_surface_cell(crystal, hkl, search=6, angle_tol=1.0, tol=1e-8):
    """
    Brute-force search for a near-rectangular in-plane surface cell.

    Enumerates lattice vectors lying in the (hkl) plane (up to
    -search ... +search) and checks every independent pair, looking
    for the one closest to 90 degrees between v1 and v2. Among pairs
    within `angle_tol` degrees of perfectly rectangular, the smallest
    area (|v1 x v2|) is preferred, to keep the resulting surface cell
    as small as possible. If no pair is found within `angle_tol`, the
    closest one overall is returned instead (with a warning).

    Parameters
    ----------
    crystal : Crystal
        Crystal object containing the unit cell.

    hkl : tuple(int, int, int)
        Miller indices.

    search : int
        Search integer combinations from -search ... +search.

    angle_tol : float
        Acceptable deviation from 90 degrees (in degrees) between v1
        and v2.

    tol : float
        Tolerance for plane membership / vector independence.

    Returns
    -------
    PlaneBasis
        Information describing the (near-)rectangular 2D lattice of
        the Miller plane.
    """

    lattice = crystal.unitcell

    normal = miller_normal(lattice, hkl)

    candidates = _in_plane_lattice_vectors(lattice, normal, search=search, tol=tol)

    if len(candidates) < 2:
        raise RuntimeError("Could not find enough in-plane vectors.")

    n = len(candidates)

    best_within_tol = None
    best_within_tol_area = None

    best_overall = None
    best_overall_deviation = None

    # Brute-force scan over all independent pairs of in-plane
    # candidate vectors.
    for a in range(n):

        c1, v1 = candidates[a]
        norm1 = np.linalg.norm(v1)

        for b in range(a + 1, n):

            c2, v2 = candidates[b]

            cross_vec = np.cross(v1, v2)
            area = np.linalg.norm(cross_vec)

            if area < tol:
                continue  # not independent (parallel/collinear)

            norm2 = np.linalg.norm(v2)

            cos_angle = np.dot(v1, v2) / (norm1 * norm2)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

            deviation = abs(angle - 90.0)

            if best_overall_deviation is None or deviation < best_overall_deviation:
                best_overall_deviation = deviation
                best_overall = (c1, v1, c2, v2)

            if deviation <= angle_tol:
                if best_within_tol_area is None or area < best_within_tol_area:
                    best_within_tol_area = area
                    best_within_tol = (c1, v1, c2, v2)

    if best_within_tol is not None:
        c1, v1, c2, v2 = best_within_tol

    elif best_overall is not None:
        c1, v1, c2, v2 = best_overall

        print(
            f"Warning: no rectangular cell found for {hkl} within "
            f"angle_tol={angle_tol} deg (search={search}). Using "
            f"closest match, deviating {best_overall_deviation:.3f} "
            f"deg from 90."
        )

    else:
        raise RuntimeError("Could not find two independent in-plane vectors.")

    return PlaneBasis(
        hkl=hkl,
        normal=normal,
        coeff1=c1,
        coeff2=c2,
        v1=v1,
        v2=v2,
    )


def make_surface_cell(crystal, hkl, search=6, angle_tol=1.0, tol=1e-8, rectangular=True):
    """
    Construct a full 3D periodic surface cell adapted to a Miller
    plane (hkl), suitable for building slabs.

    v1 and v2 span the exposed (hkl) surface (found via brute-force
    search, optionally optimized towards a rectangular in-plane cell
    via `find_rectangular_surface_cell`). v3 is found via a separate
    brute-force search: the lattice vector with the smallest positive
    projection onto the surface normal, among vectors NOT lying in
    the plane -- i.e. the smallest periodic repeat distance along the
    stacking direction (the interlayer spacing). When several vectors
    share (numerically) the same projection, the shortest overall
    vector is preferred, to keep the cell as unskewed as possible.

    Parameters
    ----------
    crystal : Crystal
        Crystal object containing the unit cell.

    hkl : tuple(int, int, int)
        Miller indices.

    search : int
        Search integer combinations from -search ... +search, used
        both for the in-plane search and the out-of-plane search.

    angle_tol : float
        Passed to find_rectangular_surface_cell (ignored if
        rectangular=False).

    tol : float
        Numerical tolerance for plane membership / independence.

    rectangular : bool
        If True (default), v1/v2 come from find_rectangular_surface_cell.
        If False, v1/v2 come from the minimal (possibly oblique) pair
        given by find_plane_vectors.

    Returns
    -------
    SurfaceCell
        v1, v2 span the surface plane; v3 closes the periodic cell
        along the stacking direction.
    """

    lattice = crystal.unitcell

    normal = miller_normal(lattice, hkl)

    if rectangular:
        basis = find_rectangular_surface_cell(
            crystal, hkl, search=search, angle_tol=angle_tol, tol=tol
        )
    else:
        basis = find_plane_vectors(crystal, hkl, search=search, tol=tol)

    # --------------------------------------------------------------
    # Brute-force search for v3: among all lattice vectors NOT lying
    # in the plane, find the smallest positive projection onto the
    # normal (the smallest out-of-plane periodic repeat). Ties are
    # broken by preferring the shortest overall vector, to minimize
    # how much the cell is skewed away from the surface normal.
    # --------------------------------------------------------------

    best = None
    best_score = None

    for i in range(-search, search + 1):
        for j in range(-search, search + 1):
            for k in range(-search, search + 1):

                if i == 0 and j == 0 and k == 0:
                    continue

                coeff = np.array([i, j, k])

                vec = coeff @ lattice

                projection = np.dot(vec, normal)

                if projection <= tol:
                    continue  # in-plane, or pointing the wrong way

                # Round to avoid floating point noise splitting what
                # should be tied projection distances.
                score = (round(projection, 8), np.linalg.norm(vec))

                if best_score is None or score < best_score:
                    best_score = score
                    best = (coeff, vec)

    if best is None:
        raise RuntimeError(
            f"Could not find an out-of-plane vector for {hkl} "
            f"(try increasing search)."
        )

    coeff3, v3 = best

    return SurfaceCell(
        basis=basis,
        v1=basis.v1,
        v2=basis.v2,
        v3=v3,
        coeff3=coeff3,
    )

def build_surface_unit_cell(crystal, surface, tol=1e-6):
    """
    Populate the surface-adapted cell (v1, v2, v3) from a SurfaceCell
    with atoms, by brute-force replication of the original unit cell.

    The new cell (v1, v2, v3) is an integer combination of the
    original lattice: V = C @ unitcell, where C is the 3x3 integer
    matrix formed from the coefficient rows stored on `surface`. This
    function enumerates enough replicas of the original unit cell
    (each carrying a full copy of the basis atoms) to cover the new
    cell, converts every candidate atom to fractional coordinates of
    the NEW cell, and keeps exactly the ones that fall inside
    [0, 1) along all three new axes. The result is exactly
    len(basis) * abs(det(C)) atoms -- one consistent filling of the
    surface-oriented cell.

    Parameters
    ----------
    crystal : Crystal
        Original bulk crystal (unrotated, unmodified unitcell/basis).

    surface : SurfaceCell
        Output of make_surface_cell.

    tol : float
        Tolerance for the "inside the new cell" fractional-coordinate
        test.

    Returns
    -------
    cell : (3,3) ndarray
        The new cell, rows = [v1, v2, v3].

    positions : (N,3) ndarray
        Cartesian atom positions inside the new cell.
    """

    lattice = crystal.unitcell
    basis = crystal.basis

    cell = np.array([surface.v1, surface.v2, surface.v3])

    C = np.array([
        surface.basis.coeff1,
        surface.basis.coeff2,
        surface.coeff3,
    ])

    # Bounding range of original-cell replicas needed to cover the
    # new cell: examine all 8 corners of the new cell, expressed as
    # multiples of the original lattice vectors.
    corners = []
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                corners.append(x*C[0] + y*C[1] + z*C[2])
    corners = np.array(corners)

    lo = np.floor(corners.min(axis=0)).astype(int) - 1
    hi = np.ceil(corners.max(axis=0)).astype(int) + 1

    inv_cell = np.linalg.inv(cell)

    positions = []

    for i in range(lo[0], hi[0] + 1):
        for j in range(lo[1], hi[1] + 1):
            for k in range(lo[2], hi[2] + 1):

                shift = np.array([i, j, k], dtype=float)

                for atom in basis:

                    cart = (atom + shift) @ lattice

                    frac_new = cart @ inv_cell

                    if np.all(frac_new > -tol) and np.all(frac_new < 1 - tol):
                        positions.append(cart)

    positions = np.array(positions)

    expected = len(basis) * abs(int(round(np.linalg.det(C))))

    if len(positions) != expected:
        print(
            f"Warning: expected {expected} atoms in the surface cell "
            f"for {surface.basis.hkl}, found {len(positions)}. "
            f"Consider adjusting tol."
        )

    return cell, positions


def assemble_slab(cell, unit_positions, normal, n1, n2, n3, vacuum=0.0):
    """
    Tile a surface unit cell into a finite slab and orient it with
    the surface normal along +z.

    Parameters
    ----------
    cell : (3,3) ndarray
        Surface-adapted cell vectors, rows = [v1, v2, v3], as
        returned by build_surface_unit_cell.

    unit_positions : (M,3) ndarray
        Cartesian atom positions within one copy of `cell`.

    normal : (3,) ndarray
        Unit normal of the exposed (hkl) plane, in the SAME
        (original, unrotated) frame as `cell` / `unit_positions`.

    n1, n2 : int
        Number of repeats along v1, v2 (lateral slab size).

    n3 : int
        Number of repeats along v3 (slab thickness). Note that one
        repeat of v3 may already bundle several atomic planes,
        depending on (hkl) and the lattice, so n3 is not necessarily
        a literal atomic-layer count.

    vacuum : float
        Vacuum gap added above the slab, along z (same units as the
        lattice constant).

    Returns
    -------
    supercell : (3,3) ndarray
        Final orthogonal box: [[Lx,0,0], [0,Ly,0], [0,0,Lz]].

    positions : (N,3) ndarray
        Cartesian positions of all atoms: (hkl) normal along +z, v1
        along +x, v2 along +y, wrapped into the box in x/y, and
        shifted so the slab starts at z=0 with `vacuum` above it.
    """

    v1, v2, v3 = cell

    # --- Tile the surface unit cell n1 x n2 x n3 times ---
    tiles = []

    for i in range(n1):
        for j in range(n2):
            for k in range(n3):

                shift = i*v1 + j*v2 + k*v3

                tiles.append(unit_positions + shift)

    positions = np.concatenate(tiles, axis=0)

    # --- Rotate so the surface normal points along +z ---
    R1 = rotation_to_z(normal)

    positions = positions @ R1.T
    v1r = v1 @ R1.T
    v2r = v2 @ R1.T

    # --- Rotate about z so v1 aligns with +x (axis-aligned box) ---
    theta = np.arctan2(v1r[1], v1r[0])

    c, s = np.cos(-theta), np.sin(-theta)

    R2 = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])

    positions = positions @ R2.T
    v1r = v1r @ R2.T
    v2r = v2r @ R2.T

    # Box lengths (v1r should now be ~[|v1|, 0, 0], v2r ~[0, ±|v2|, 0]
    # since v1 ⟂ v2 thanks to find_rectangular_surface_cell)
    Lx = n1 * v1r[0]
    Ly = n2 * abs(v2r[1])

    z = positions[:, 2]
    z_min = z.min()

    # Shift slab to start at z = 0
    positions[:, 2] -= z_min

    Lz = (z.max() - z_min) + vacuum

    # Box is axis-aligned and orthogonal, so a plain modulo correctly
    # wraps atoms into [0, Lx) x [0, Ly) -- no triclinic wrap needed.
    positions[:, 0] %= Lx
    positions[:, 1] %= Ly

    supercell = np.array([
        [Lx, 0.0, 0.0],
        [0.0, Ly, 0.0],
        [0.0, 0.0, Lz],
    ])

    return supercell, positions


def orient_crystal(crystal, hkl):
    """
    Rotate a crystal so that the Miller plane (hkl)
    becomes parallel to the xy-plane.

    Returns
    -------
    rotated : Crystal
        Rotated crystal.
    R : ndarray
        Rotation matrix.
    """

    normal = miller_normal(crystal.unitcell, hkl)

    R = rotation_to_z(normal)

    rotate_crystal(crystal, R)

    return crystal, R

def rotation_between_vectors(v_from, v_to):
    """
    Return a rotation matrix that rotates v_from onto v_to.

    Parameters
    ----------
    v_from : array-like (3,)
        Initial vector.

    v_to : array-like (3,)
        Target vector.

    Returns
    -------
    R : (3,3) ndarray
        Rotation matrix.
    """

    v_from = np.asarray(v_from, dtype=float)
    v_to = np.asarray(v_to, dtype=float)

    # Normalize
    v_from /= np.linalg.norm(v_from)
    v_to /= np.linalg.norm(v_to)

    rotation, rmsd = Rotation.align_vectors(
        [v_to],
        [v_from]
    )

    return rotation.as_matrix()

def rotation_to_z(normal):
    """
    Rotate a vector onto the z-axis.
    """

    return rotation_between_vectors(
        normal,
        np.array([0.0, 0.0, 1.0])
    )


def rotate_crystal(crystal, R):
    """
    Rotate the current simulation cell and particle positions.

    The original unit cell is left unchanged.
    """

    crystal.positions = crystal.positions @ R.T
    crystal.supercell = crystal.supercell @ R.T

    return crystal


def print_lattice_info(lattice):
    """
    Print the lattice vectors and the angles between them.
    """

    print("\nLattice vectors")

    for i, vec in enumerate(lattice):
        print(f"a{i+1} =", vec)

    lengths = np.linalg.norm(lattice, axis=1)

    print("\nLengths")

    print(lengths)

    print("\nAngles")

    for i in range(3):
        for j in range(i+1, 3):

            angle = np.degrees(
                np.arccos(
                    np.clip(
                        np.dot(lattice[i], lattice[j]) /
                        (lengths[i]*lengths[j]),
                        -1,
                        1,
                    )
                )
            )

            print(f"a{i+1}-a{j+1}: {angle:.6f}°")
