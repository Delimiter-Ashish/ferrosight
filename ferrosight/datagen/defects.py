"""Turn a design into an as-built structure by injecting known discrepancies."""
from __future__ import annotations

import copy
import random

from . import sections as S
from .structure import Component, Design

DEFECT_TYPES = ["missing", "wrong_section", "misplaced", "extra"]
HORIZONTAL = ("girder", "beam", "joist")


def _retop(c: Component, old: str, new: str) -> None:
    """Keep top of steel where it was when the section depth changes."""
    if c.kind in HORIZONTAL:
        dz = S.depth(old) / 2 - S.depth(new) / 2
        c.start[2] = round(c.start[2] + dz, 4)
        c.end[2] = round(c.end[2] + dz, 4)


def inject_defects(design: Design, seed: int, n_defects: int | None = None):
    """Returns (as_built_components, defects). Each component gets at most one defect."""
    rng = random.Random(seed * 7919 + 13)
    if n_defects is None:
        n_defects = rng.choice([0, 1, 2, 2, 3, 3, 4, 5])   # some scenes stay clean
    built = {c.id: copy.deepcopy(c) for c in design.components}
    defects: list[dict] = []
    touched: set[str] = set()

    def pick(kinds):
        pool = [c for c in design.components if c.kind in kinds and c.id not in touched]
        return rng.choice(pool) if pool else None

    def log(dtype, cid, desc, design_state, built_state):
        defects.append(dict(defect_id=f"D{len(defects)+1:02d}", type=dtype, component_id=cid,
                            description=desc, design=design_state, as_built=built_state))
        touched.add(cid)

    for _ in range(n_defects):
        dtype = rng.choice(DEFECT_TYPES)

        if dtype == "missing":
            c = pick(("girder", "beam", "joist", "brace", "baseplate"))
            if not c:
                continue
            del built[c.id]
            log("missing", c.id, f"{c.kind} {c.section} at {c.grid} (level {c.level}) not installed",
                dict(section=c.section, start=c.start, end=c.end), None)

        elif dtype == "wrong_section":
            c = pick(("girder", "beam", "joist", "brace"))
            if not c:
                continue
            options = S.visibly_different(c.section)
            if c.kind != "brace":      # keep beams as wide-flange so the swap is plausible
                options = [o for o in options if S.SECTIONS[o]["shape"] == "I"]
            if not options:
                continue
            new = rng.choice(options)
            b = built[c.id]
            _retop(b, c.section, new)
            b.section = new
            log("wrong_section", c.id, f"{c.kind} at {c.grid} (level {c.level}) is {new}, drawings call for {c.section}",
                dict(section=c.section), dict(section=new, start=b.start, end=b.end))

        elif dtype == "misplaced":
            c = pick(("girder", "beam", "joist", "brace"))
            if not c:
                continue
            b = built[c.id]
            if c.kind == "brace":
                b.start[2], b.end[2] = b.end[2], b.start[2]
                desc = f"brace at {c.grid} (level {c.level}) installed on the opposite diagonal"
            elif c.kind == "joist" and rng.random() < 0.5:
                dx = rng.choice([-1, 1]) * rng.uniform(0.8, 1.4)
                b.start[0] = round(b.start[0] + dx, 3)
                b.end[0] = round(b.end[0] + dx, 3)
                desc = f"joist at {c.grid} (level {c.level}) shifted {abs(dx):.2f} m off its setting-out line"
            else:
                dz = rng.uniform(0.45, 0.9)
                b.start[2] = round(b.start[2] - dz, 3)
                b.end[2] = round(b.end[2] - dz, 3)
                desc = f"{c.kind} at {c.grid} (level {c.level}) set {dz:.2f} m below design elevation"
            log("misplaced", c.id, desc, dict(start=c.start, end=c.end), dict(start=b.start, end=b.end))

        elif dtype == "extra":
            src = pick(("joist", "beam"))
            if not src:
                continue
            e = copy.deepcopy(src)
            lo, hi = design.grid_x[0], design.grid_x[-1]
            taken = {round(c.start[0], 1) for c in built.values()
                     if c.kind in ("beam", "joist") and c.level == src.level}
            xs = [round(lo + 0.5 * t, 1) for t in range(1, int((hi - lo) / 0.5))]
            xs = [x for x in xs if all(abs(x - t) >= 0.9 for t in taken)]
            if not xs:
                continue
            x = rng.choice(xs)
            e.id = f"X-{len(defects)+1:02d}-L{src.level}"
            e.kind = "joist"
            e.start[0] = e.end[0] = x
            e.grid = f"x={x:.1f} m / {src.grid.split('/')[-1]}"
            e.frame = ""
            built[e.id] = e
            log("extra", e.id, f"unspecified {e.section} member at {e.grid} (level {e.level}), not on drawings",
                None, dict(section=e.section, start=e.start, end=e.end))
            touched.add(src.id)

    return list(built.values()), defects
