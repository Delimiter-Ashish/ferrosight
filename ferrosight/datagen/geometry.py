"""Shared member geometry: local frame and profile outline."""
from __future__ import annotations

import numpy as np

from .sections import SECTIONS


def local_frame(start, end, up):
    """Return (x_axis, y_axis, z_axis, length). z runs along the member, y is the web/depth direction."""
    a, b, u = np.asarray(start, float), np.asarray(end, float), np.asarray(up, float)
    z = b - a
    length = float(np.linalg.norm(z))
    z = z / length
    x = np.cross(u, z)
    if np.linalg.norm(x) < 1e-6:                  # up is parallel to the axis
        x = np.cross([0.0, 1.0, 0.0], z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z, length


def placement_matrix(start, end, up) -> np.ndarray:
    x, y, z, _ = local_frame(start, end, up)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x, y, z, np.asarray(start, float)
    return m


def profile_outline(section: str) -> list[tuple[float, float]]:
    """Closed 2D outline (x = width, y = depth), counter-clockwise."""
    s = SECTIONS[section]
    hd, hb = s["d"] / 2, s["bf"] / 2
    if s["shape"] == "RECT":
        return [(-hb, -hd), (hb, -hd), (hb, hd), (-hb, hd)]
    hw, tf = s["tw"] / 2, s["tf"]
    return [(-hb, -hd), (hb, -hd), (hb, -hd + tf), (hw, -hd + tf), (hw, hd - tf), (hb, hd - tf),
            (hb, hd), (-hb, hd), (-hb, hd - tf), (-hw, hd - tf), (-hw, -hd + tf), (-hb, -hd + tf)]
