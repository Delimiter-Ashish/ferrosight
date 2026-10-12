"""Linework extraction: column squares, solid/dashed orthogonal members, diagonal members.

Works on the binarised sheet only (no text). Also produces a text-only image, with the
linework erased, which is what the OCR stage reads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

INK_THRESHOLD = 90        # darker than this = drawn in ink (gridlines and level lines are lighter)
WEAK_THRESHOLD = 135      # faded / blurred line pixels; accepted only as part of a line that has real ink
MIN_RUN = 15              # shortest straight run that counts as linework rather than a text stroke


@dataclass
class Segment:
    x0: float
    y0: float
    x1: float
    y1: float
    thickness: float
    dashed: bool = False
    orient: str = "h"     # h | v | d

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.y1 - self.y0)

    @property
    def mid(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def distance_to(self, px: float, py: float) -> tuple[float, float]:
        """(perpendicular distance, position along the segment as a 0..1 fraction)."""
        dx, dy = self.x1 - self.x0, self.y1 - self.y0
        t = ((px - self.x0) * dx + (py - self.y0) * dy) / max(1e-9, dx * dx + dy * dy)
        qx, qy = self.x0 + t * dx, self.y0 + t * dy
        return math.hypot(px - qx, py - qy), t


@dataclass
class Square:
    cx: float
    cy: float
    size: float


@dataclass
class Linework:
    squares: list
    horizontal: list
    vertical: list
    diagonal: list
    text_image: np.ndarray      # grayscale sheet with linework erased, for OCR
    ink: np.ndarray


def _components(mask: np.ndarray):
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        yield int(x), int(y), int(w), int(h), int(area), cent[i]


def find_squares(ink: np.ndarray) -> tuple[list[Square], np.ndarray]:
    """Solid squares (columns on plan). Found as blobs far thicker than any line."""
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    core = (dist >= 4.5).astype(np.uint8)
    squares, mask = [], np.zeros_like(ink)
    for x, y, w, h, _a, c in _components(core):
        half = float(dist[y:y + h, x:x + w].max())
        if half < 6.0:
            continue                                  # a junction of two lines, not a drawn square
        if max(w, h) > 2.2 * min(w, h) + 2:
            continue                                  # elongated: a very thick line, not a square
        s = Square(float(c[0]), float(c[1]), 2 * half)
        squares.append(s)
        p = int(round(half)) + 3
        cv2.rectangle(mask, (int(s.cx) - p, int(s.cy) - p), (int(s.cx) + p, int(s.cy) + p), 1, -1)
    return squares, mask


def _runs(ink: np.ndarray, orient: str) -> np.ndarray:
    k = (MIN_RUN, 2) if orient == "h" else (2, MIN_RUN)
    return cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, k))


def _line_runs(strong: np.ndarray, weak: np.ndarray, orient: str, exclude: np.ndarray) -> np.ndarray:
    """Straight runs found with a lenient threshold (so a faded or blurred thin line stays in one
    piece), kept only where the run also contains real ink. Light-grey gridlines never do."""
    runs = _runs(weak, orient) * (1 - exclude)
    n, lab = cv2.connectedComponents(runs, connectivity=8)
    if n <= 1:
        return runs
    area = np.bincount(lab.ravel(), minlength=n)
    dark = np.bincount(lab.ravel(), weights=strong.ravel().astype(np.float64), minlength=n)
    keep = (dark / np.maximum(area, 1)) >= 0.25
    keep[0] = False
    return keep[lab].astype(np.uint8)


def _orthogonal(mask: np.ndarray, orient: str) -> list[Segment]:
    """Turn a run mask into solid segments, then chain collinear short runs into dashed segments."""
    pieces = []
    for x, y, w, h, area, c in _components(mask):
        length = w if orient == "h" else h
        thick = area / max(1, length)          # mean width: robust to a plate or knob at one end
        if thick > 12 or length < MIN_RUN:
            continue
        along0, along1 = (x, x + w) if orient == "h" else (y, y + h)
        across = float(c[1]) if orient == "h" else float(c[0])
        pieces.append([along0, along1, across, thick])
    # rejoin a line that a crossing light line (a gridline drawn on top) has cut by a pixel or two
    pieces.sort(key=lambda p: (round(p[2] / 3), p[0]))
    joined: list[list[float]] = []
    for p in pieces:
        q = joined[-1] if joined else None
        if q and abs(q[2] - p[2]) <= 2 and 0 <= p[0] - q[1] <= 3:
            la, lb = q[1] - q[0], p[1] - p[0]
            joined[-1] = [q[0], p[1], (q[2] * la + p[2] * lb) / (la + lb), (q[3] * la + p[3] * lb) / (la + lb)]
        else:
            joined.append(list(p))
    pieces = joined
    used, out = [False] * len(pieces), []
    for i, p in enumerate(pieces):
        if used[i]:
            continue
        chain, used[i] = [p], True
        if p[1] - p[0] <= 60:                          # dash-sized: try to extend along the line
            grew = True
            while grew:
                grew = False
                for j, q in enumerate(pieces):
                    if used[j] or q[1] - q[0] > 60 or abs(q[2] - chain[-1][2]) > 2.5:
                        continue
                    gap = q[0] - chain[-1][1]
                    if 0 < gap <= 0.9 * max(q[1] - q[0], chain[-1][1] - chain[-1][0]) + 6:
                        chain.append(q), used.__setitem__(j, True)
                        grew = True
                        break
        a0, a1 = chain[0][0], chain[-1][1]
        across = float(np.mean([c[2] for c in chain]))
        thick = float(np.mean([c[3] for c in chain]))
        dashed = len(chain) >= 3
        if len(chain) == 2 or (len(chain) == 1 and a1 - a0 < 2 * MIN_RUN):
            continue                                   # stray fragment
        if orient == "h":
            out.append(Segment(a0, across, a1, across, thick, dashed, "h"))
        else:
            out.append(Segment(across, a0, across, a1, thick, dashed, "v"))
    return out


def _diagonals(ink: np.ndarray, taken: np.ndarray, weak: np.ndarray | None = None) -> tuple[list[Segment], np.ndarray]:
    base = ink if weak is None else weak      # lenient threshold keeps a faded brace in one piece
    rest = cv2.subtract(base, cv2.dilate(taken, np.ones((5, 5), np.uint8)))
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    segs, mask = [], np.zeros_like(ink)
    n, lab, stats, _c = cv2.connectedComponentsWithStats(rest, connectivity=8)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if min(w, h) < 20 or area < 80:
            continue
        if ink[lab == i].mean() < 0.25:
            continue                                  # grey line, not drawn in ink
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        pts = np.c_[xs + x, ys + y].astype(np.float32)
        (cx, cy), (rw, rh), _ang = cv2.minAreaRect(pts)
        length, thick = max(rw, rh), min(rw, rh)
        if length < 40 or thick > 10 or length < 8 * thick:
            continue
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        t = (pts[:, 0] - x0) * vx + (pts[:, 1] - y0) * vy
        a, b = (x0 + t.min() * vx, y0 + t.min() * vy), (x0 + t.max() * vx, y0 + t.max() * vy)
        if a[0] > b[0]:
            a, b = b, a
        segs.append(Segment(float(a[0]), float(a[1]), float(b[0]), float(b[1]), float(thick), False, "d"))
        mask[lab == i] = 1
    return segs, mask


def extract(gray: np.ndarray, region: tuple[int, int, int, int] | None = None) -> Linework:
    """`region` (x0, y0, x1, y1) limits linework detection, e.g. to exclude the title block."""
    ink = (gray < INK_THRESHOLD).astype(np.uint8)
    work = ink.copy()
    if region is not None:
        x0, y0, x1, y1 = region
        keep = np.zeros_like(ink)
        keep[y0:y1, x0:x1] = 1
        work = work * keep
    weak = (gray < WEAK_THRESHOLD).astype(np.uint8)
    if region is not None:
        weak = weak * keep
    squares, sq_mask = find_squares(work)
    h_mask = _line_runs(work, weak, "h", sq_mask)
    v_mask = _line_runs(work, weak, "v", sq_mask)
    horizontal = _orthogonal(h_mask, "h")
    vertical = _orthogonal(v_mask, "v")
    taken = cv2.bitwise_or(cv2.bitwise_or(h_mask, v_mask), sq_mask)
    diagonal, d_mask = _diagonals(work, taken, weak)

    text = gray.copy()          # grey levels are left alone: thin text strokes are never fully dark
    erase = cv2.dilate(cv2.bitwise_or(taken, d_mask), np.ones((3, 3), np.uint8))
    text[erase > 0] = 255
    return Linework(squares, horizontal, vertical, diagonal, text, ink)
