"""Parametric steel-frame generator. Produces the *design* (what the drawings say)."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import sections as S

LETTERS = "ABCDEFGH"


@dataclass
class Component:
    id: str
    kind: str            # column | girder | beam | joist | brace | baseplate
    section: str
    start: list[float]   # member axis start, metres
    end: list[float]     # member axis end, metres
    up: list[float]      # reference vector fixing the web orientation
    level: int           # storey the member belongs to (0 = foundation)
    grid: str            # human-readable location, e.g. "A/1-2"
    frame: str = ""      # gridline whose elevation shows this member ("" if none)

    def length(self) -> float:
        return sum((b - a) ** 2 for a, b in zip(self.start, self.end)) ** 0.5


@dataclass
class Design:
    name: str
    seed: int
    grid_x: list[float]          # positions of numbered gridlines (1, 2, 3 ...)
    grid_y: list[float]          # positions of lettered gridlines (A, B, C ...)
    levels: list[float]          # top-of-steel elevation of each storey
    components: list[Component] = field(default_factory=list)

    @property
    def x_labels(self) -> list[str]:
        return [str(i + 1) for i in range(len(self.grid_x))]

    @property
    def y_labels(self) -> list[str]:
        return [LETTERS[j] for j in range(len(self.grid_y))]

    def by_id(self) -> dict[str, Component]:
        return {c.id: c for c in self.components}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["x_labels"], d["y_labels"] = self.x_labels, self.y_labels
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1))

    @staticmethod
    def load(path: str | Path) -> "Design":
        d = json.loads(Path(path).read_text())
        comps = [Component(**c) for c in d["components"]]
        return Design(d["name"], d["seed"], d["grid_x"], d["grid_y"], d["levels"], comps)


def _r(v: float) -> float:
    return round(v, 4)


def generate_design(seed: int, name: str | None = None) -> Design:
    rng = random.Random(seed)
    nx = rng.randint(2, 4)                  # bays along X
    ny = rng.randint(1, 3)                  # bays along Y
    storeys = rng.randint(1, 3)

    gx = [0.0]
    for _ in range(nx):
        gx.append(_r(gx[-1] + rng.choice([5.0, 6.0, 7.5, 9.0])))
    gy = [0.0]
    for _ in range(ny):
        gy.append(_r(gy[-1] + rng.choice([5.0, 6.0, 7.5])))
    h = rng.choice([3.6, 4.0, 4.5])
    levels = [_r(h * (k + 1)) for k in range(storeys)]

    col_sec = rng.choice(S.COLUMN_SECTIONS)
    gir_sec = rng.choice(S.GIRDER_SECTIONS)
    beam_sec = rng.choice(S.BEAM_SECTIONS)
    joist_sec = rng.choice(S.JOIST_SECTIONS)
    brace_sec = rng.choice(S.BRACE_SECTIONS)
    n_infill = rng.randint(0, 2)
    half_col = S.depth(col_sec) / 2

    d = Design(name or f"FS-{seed:04d}", seed, gx, gy, levels)
    xl, yl = d.x_labels, d.y_labels
    add = d.components.append

    # base plates + columns
    for j, y in enumerate(gy):
        for i, x in enumerate(gx):
            loc = f"{yl[j]}/{xl[i]}"
            add(Component(f"BP-{yl[j]}{xl[i]}", "baseplate", S.BASEPLATE_SECTION,
                          [x, y, -S.BASEPLATE_THICKNESS], [x, y, 0.0], [0, 1, 0], 0, loc))
            for k, z in enumerate(levels):
                z0 = 0.0 if k == 0 else levels[k - 1]
                add(Component(f"C-{yl[j]}{xl[i]}-L{k+1}", "column", col_sec,
                              [x, y, z0], [x, y, z], [0, 1, 0], k + 1, loc))

    for k, z in enumerate(levels):
        lv = k + 1
        # girders along X on every lettered gridline
        zc = _r(z - S.depth(gir_sec) / 2)
        for j, y in enumerate(gy):
            for i in range(nx):
                add(Component(f"G-{yl[j]}{xl[i]}{xl[i+1]}-L{lv}", "girder", gir_sec,
                              [_r(gx[i] + half_col), y, zc], [_r(gx[i+1] - half_col), y, zc],
                              [0, 0, 1], lv, f"{yl[j]}/{xl[i]}-{xl[i+1]}", frame=yl[j]))
        # beams along Y on every numbered gridline
        zc = _r(z - S.depth(beam_sec) / 2)
        for i, x in enumerate(gx):
            for j in range(ny):
                add(Component(f"B-{xl[i]}{yl[j]}{yl[j+1]}-L{lv}", "beam", beam_sec,
                              [x, _r(gy[j] + half_col), zc], [x, _r(gy[j+1] - half_col), zc],
                              [0, 0, 1], lv, f"{xl[i]}/{yl[j]}-{yl[j+1]}", frame=xl[i]))
        # infill joists along Y inside each X bay
        zc = _r(z - S.depth(joist_sec) / 2)
        for i in range(nx):
            for m in range(n_infill):
                x = _r(gx[i] + (gx[i+1] - gx[i]) * (m + 1) / (n_infill + 1))
                for j in range(ny):
                    add(Component(f"J-{xl[i]}{xl[i+1]}.{m+1}{yl[j]}{yl[j+1]}-L{lv}", "joist", joist_sec,
                                  [x, _r(gy[j] + 0.1), zc], [x, _r(gy[j+1] - 0.1), zc],
                                  [0, 0, 1], lv, f"{xl[i]}-{xl[i+1]}/{yl[j]}-{yl[j+1]}"))

    # vertical bracing: one braced bay in each perimeter frame, every storey
    frames = [("X", 0), ("X", ny), ("Y", 0), ("Y", nx)]
    rng.shuffle(frames)
    for axis, idx in frames[: rng.randint(2, 4)]:
        bay = rng.randrange(nx if axis == "X" else ny)
        for k, z in enumerate(levels):
            z0 = (0.0 if k == 0 else levels[k - 1]) + 0.15
            z1 = z - 0.45
            if rng.random() < 0.5:
                z0, z1 = z1, z0
            if axis == "X":
                y = gy[idx]
                a, b = [_r(gx[bay] + 0.2), y, _r(z0)], [_r(gx[bay+1] - 0.2), y, _r(z1)]
                cid = f"BR-{yl[idx]}{xl[bay]}{xl[bay+1]}-L{k+1}"
                loc, frame = f"{yl[idx]}/{xl[bay]}-{xl[bay+1]}", yl[idx]
            else:
                x = gx[idx]
                a, b = [x, _r(gy[bay] + 0.2), _r(z0)], [x, _r(gy[bay+1] - 0.2), _r(z1)]
                cid = f"BR-{xl[idx]}{yl[bay]}{yl[bay+1]}-L{k+1}"
                loc, frame = f"{xl[idx]}/{yl[bay]}-{yl[bay+1]}", xl[idx]
            add(Component(cid, "brace", brace_sec, a, b, [0, 0, 1], k + 1, loc, frame=frame))
    return d
