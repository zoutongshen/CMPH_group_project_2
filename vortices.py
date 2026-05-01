"""
Vortex detection on the 2D XY lattice.

For each 2x2 plaquette of the periodic lattice we compute the discrete
winding number: the sum of the four nearest-neighbour angle differences
around the plaquette, each wrapped into (-pi, pi], divided by 2 pi. The
result is +1 at a vortex, -1 at an anti-vortex, and 0 elsewhere. On a
periodic lattice the topological charge sums to zero, so every simulation
carries equal numbers of vortices and anti-vortices.

Plaquette (r, c) is the square with its top-left corner at site (r, c) and
corners (r, c), (r, c+1), (r+1, c+1), (r+1, c) under periodic boundary
conditions.

Author: CMPH 2026 Project 2
Date: April 2026
"""

from typing import Tuple

import numpy as np


def _wrap_to_pi(differences: np.ndarray) -> np.ndarray:
    """Wrap angle differences into the half-open interval [-pi, pi)."""
    return np.mod(differences + np.pi, 2.0 * np.pi) - np.pi


def plaquette_charges(angles: np.ndarray) -> np.ndarray:
    """
    Topological charge of every plaquette of the lattice.

    Walks each 2x2 plaquette top_left -> top_right -> bottom_right ->
    bottom_left -> top_left (under periodic boundary conditions) and sums
    the four nearest-neighbour angle differences, each wrapped into
    (-pi, pi]. The closed-loop sum divided by 2 pi rounds to the integer
    winding number of the plaquette.

    Args:
        angles: Array of shape (size, size) with spin angles

    Returns:
        Integer array of shape (size, size) whose (r, c) entry is the
        topological charge (+1 / -1 / 0) of the plaquette with top-left
        corner at (r, c).
    """
    # Build the four corners of every plaquette as full lattice arrays.
    # np.roll(..., -1, axis=1) moves each column one step to the left,
    # i.e. corner.[r, c] = angles[r, (c+1) % size] under PBC.
    top_left = angles
    top_right = np.roll(angles, -1, axis=1)
    bottom_right = np.roll(np.roll(angles, -1, axis=0), -1, axis=1)
    bottom_left = np.roll(angles, -1, axis=0)

    # Each leg is the nearest-neighbour angle difference along one edge of
    # the plaquette, wrapped into (-pi, pi] to remove the 2-pi gauge freedom.
    winding = (
        _wrap_to_pi(top_right - top_left)         # top edge: TL -> TR
        + _wrap_to_pi(bottom_right - top_right)   # right edge: TR -> BR
        + _wrap_to_pi(bottom_left - bottom_right) # bottom edge: BR -> BL
        + _wrap_to_pi(top_left - bottom_left)     # left edge: BL -> TL
    )
    return np.round(winding / (2.0 * np.pi)).astype(int)


def count_vortices(angles: np.ndarray) -> Tuple[int, int]:
    """
    Number of vortices and anti-vortices in the configuration.

    Args:
        angles: Array of shape (size, size) with spin angles

    Returns:
        Tuple (n_vortices, n_antivortices).
    """
    charges = plaquette_charges(angles)
    return int(np.sum(charges == 1)), int(np.sum(charges == -1))


def vortex_locations(angles: np.ndarray
                     ) -> Tuple[Tuple[np.ndarray, np.ndarray],
                                Tuple[np.ndarray, np.ndarray]]:
    """
    Plaquette-centre coordinates of the vortices and anti-vortices.

    A plaquette centre sits at (r + 0.5, c + 0.5) so that it can be drawn
    directly on top of an imshow() of the spin field.

    Args:
        angles: Array of shape (size, size) with spin angles

    Returns:
        ((vortex_rows, vortex_cols), (antivortex_rows, antivortex_cols)),
        each as arrays of floats.
    """
    charges = plaquette_charges(angles)
    vortex_rows, vortex_cols = np.where(charges == 1)
    antivortex_rows, antivortex_cols = np.where(charges == -1)
    return (
        (vortex_rows + 0.5, vortex_cols + 0.5),
        (antivortex_rows + 0.5, antivortex_cols + 0.5),
    )
