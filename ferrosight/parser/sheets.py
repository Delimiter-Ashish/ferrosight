"""Per-sheet interpretation: framing plans, frame elevations, and the notes/schedule sheet.

Everything here works in sheet pixels first and converts to metres only through the
sheet's own references (grid columns and dimension strings on plans, level annotations
and grid stations on elevations).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..datagen.structure import LETTERS
from .ocr import TextBox
from .raster import Linework, Segment
from .text import match_section, normalise, parse_number, split_tag

MIN_PIECE = 20
TAGGY = re.compile(r"(W\s?\d|HSS|PL\s?\d|\[|BR-|VB|[GBJC]-[A-H0-9])")


class ParseError(RuntimeError):
    pass


# ----------------------------------------------------------------------------- helpers
def cluster(values, tol: float) -> list[float]:
    out: list[list[float]] = []
    for v in sorted(values):
        if out and v - out[-1][-1] <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return [float(np.mean(c)) for c in out]


def nearest(values, v):
    i = int(np.argmin([abs(x - v) for x in values]))
    return i, abs(values[i] - v)


class Axis:
    """Pixel <-> metre mapping that is exact at reference positions (gridlines) and uses
    the sheet scale outside them."""

    def __init__(self, px: list[float], m: list[float], scale: float):
        order = np.argsort(px)
        self.px, self.m = np.asarray(px, float)[order], np.asarray(m, float)[order]
        self.sign = 1.0 if len(self.m) < 2 or self.m[-1] >= self.m[0] else -1.0
        self.scale = scale

    def __call__(self, p: float) -> float:
        if p < self.px[0]:
            return float(self.m[0] + self.sign * (p - self.px[0]) / self.scale)
        if p > self.px[-1]:
            return float(self.m[-1] + self.sign * (p - self.px[-1]) / self.scale)
        return float(np.interp(p, self.px, self.m))


@dataclass
class Item:
    """One drawn member found on a sheet, before it is merged into the design graph."""
    kind: str
    level: int
    key: tuple                      # location key shared across sheets, e.g. ("girder", "A", 0, 1, level)
    sheet: str
    seg: Segment | None = None
    texts: list = field(default_factory=list)
    geom: dict = field(default_factory=dict)

    def tag(self, prefer=None) -> tuple[str | None, str, str | None, str]:
        """(section, how, mark, raw text) read from the attached text fragments."""
        if not self.texts:
            return None, "none", None, ""
        if self.seg is not None and self.seg.orient == "v":
            frags = sorted(self.texts, key=lambda b: -b.cy)
        else:
            frags = sorted(self.texts, key=lambda b: b.cx)
        raw = " ".join(b.text for b in frags)
        head, mark = split_tag(raw)
        section, how = match_section(head, prefer)
        return section, how, mark, raw

    def evidence(self) -> dict:
        boxes = [b.bbox for b in self.texts]
        out = dict(sheet=self.sheet)
        if self.seg is not None:
            out["line_px"] = [round(v, 1) for v in (self.seg.x0, self.seg.y0, self.seg.x1, self.seg.y1)]
        if boxes:
            out["tag_bbox_px"] = [round(min(b[0] for b in boxes), 1), round(min(b[1] for b in boxes), 1),
                                  round(max(b[2] for b in boxes), 1), round(max(b[3] for b in boxes), 1)]
        return out


def _tag_target(box: TextBox) -> str:
    t = normalise(box.text)
    if "BR-" in t or t.startswith("VB") or "HSS" in t:
        return "brace"
    if "BP-" in t or "PL5" in t or t.startswith("PL"):
        return "baseplate"
    if re.search(r"C-[A-H]\d", t):
        return "column"
    if re.search(r"(?<![A-Z])[GBJ]-", t):
        return "member"
    return "any"


def _assign(boxes: list[TextBox], items: list[Item], accept, max_dist: float) -> list[TextBox]:
    """Attach each text box to the nearest compatible drawn segment. Returns the leftovers."""
    left = []
    for b in boxes:
        best, best_d = None, max_dist
        for it in items:
            if it.seg is None or not accept(b, it):
                continue
            d, t = it.seg.distance_to(b.cx, b.cy)
            if -0.15 <= t <= 1.15 and d < best_d:
                best, best_d = it, d
        if best is None:
            left.append(b)
        else:
            best.texts.append(b)
    return left


def sheet_kind(boxes: list[TextBox], name: str) -> tuple[str, str | None]:
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        if "FRAMINGPLAN" in t:
            return "plan", None
        m = re.search(r"ELEVATIONONGRIDLINE([A-H]|\d+)", t)
        if m:
            return "elevation", m.group(1)
        if "SCHEDULE" in t:
            return "schedule", None
    # fall back to the sheet-numbering convention of structural sets: S-0xx notes, S-1xx plans, S-2xx elevations
    return {"0": "schedule", "1": "plan", "2": "elevation"}.get(name[2:3], "unknown"), None


# ----------------------------------------------------------------------------- plans
@dataclass
class PlanResult:
    sheet: str
    level: int
    tos: float
    xs_px: list
    ys_px: list            # image y of lettered gridlines, bottom (A) first
    grid_x: list
    grid_y: list
    scale: float
    items: list
    dims_read: int
    dims_inferred: int
    warnings: list


def _plan_title(boxes) -> tuple[int | None, float | None]:
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        if "FRAMINGPLAN" in t:
            lv = re.search(r"LEVEL(\d+)", t)
            el = re.search(r"EL\.?\+?([0-9O]+\.[0-9O]+)", t)
            return (int(lv.group(1)) if lv else None), (parse_number(el.group(1)) if el else None)
    return None, None


def _bays(px: list[float], dims: list[tuple[float, float]], scale_hint: float | None):
    """Bay lengths in metres from gridline pixel positions plus whatever dimension strings were read."""
    readings = {}
    for i in range(len(px) - 1):
        lo, hi = px[i], px[i + 1]
        cands = [m for c, m in dims if lo < c < hi and m and m > 0.5]
        if cands:
            readings[i] = cands[0]
    return readings


def parse_plan(sheet: str, boxes: list[TextBox], lw: Linework, level_hint: int | None = None) -> PlanResult:
    warnings = []
    level, tos = _plan_title(boxes)
    if level is None:
        level = level_hint
        warnings.append("plan title not read; level taken from sheet number")
    if len(lw.squares) < 4:
        raise ParseError(f"{sheet}: found {len(lw.squares)} column marks; cannot establish the grid")
    sq = float(np.median([s.size for s in lw.squares]))
    xs = cluster([s.cx for s in lw.squares], tol=max(4.0, sq / 3))
    ys = sorted(cluster([s.cy for s in lw.squares], tol=max(4.0, sq / 3)), reverse=True)   # bottom first
    x_lo, x_hi, y_top, y_bot = min(xs), max(xs), min(ys), max(ys)

    # dimension strings: horizontal ones above the grid, vertical ones left of it
    dim_h, dim_v = [], []
    for b in boxes:
        t = normalise(b.text).replace("O", "0").strip()
        if not re.fullmatch(r"\d{1,2}\.\d{2,3}", t):
            continue
        if b.orientation == "h" and b.cy < y_top:
            dim_h.append((b.cx, float(t)))
        elif b.orientation == "v" and b.cx < x_lo:
            dim_v.append((b.cy, float(t)))
    rx = _bays(xs, dim_h, None)
    ry = _bays(sorted(ys), dim_v, None)        # ascending image y = top bay first
    ny = len(ys) - 1
    ry = {ny - 1 - i: m for i, m in ry.items()}  # re-index bottom bay first
    ests = [(xs[i + 1] - xs[i]) / m for i, m in rx.items()] + [(ys[j] - ys[j + 1]) / m for j, m in ry.items()]
    if not ests:
        raise ParseError(f"{sheet}: no dimension strings read; drawing scale unknown")
    scale = float(np.median(ests))
    read = inferred = 0

    def bay_m(px_len, reading):
        nonlocal read, inferred
        if reading is not None and abs(px_len / scale - reading) / reading < 0.03:
            read += 1
            return reading
        inferred += 1
        return round(px_len / scale / 0.05) * 0.05

    gx, gy = [0.0], [0.0]
    for i in range(len(xs) - 1):
        gx.append(round(gx[-1] + bay_m(xs[i + 1] - xs[i], rx.get(i)), 3))
    for j in range(len(ys) - 1):
        gy.append(round(gy[-1] + bay_m(ys[j] - ys[j + 1], ry.get(j)), 3))
    if inferred:
        warnings.append(f"{inferred} bay dimension(s) not read; measured from the drawing instead")

    ax_x, ax_y = Axis(xs, gx, scale), Axis(ys, gy, scale)
    tol = max(4.0, 0.12 * scale)
    xl = [str(i + 1) for i in range(len(xs))]
    yl = [LETTERS[j] for j in range(len(ys))]
    items: list[Item] = []

    def bay_of(vals, lo, hi):
        mid = (lo + hi) / 2
        for i in range(len(vals) - 1):
            a, b = sorted((vals[i], vals[i + 1]))
            if a < mid < b:
                return i
        return None

    def split(segs, cuts, orient):
        """Plan members span between supports, so a line that runs across a gridline is two
        members whose ends merged (blur, or a thin gap filled in). Cut it at the supporting line."""
        out = []
        for s in segs:
            if s.dashed:
                out.append(s)
                continue
            a0, a1 = (s.x0, s.x1) if orient == "h" else (s.y0, s.y1)
            inner = [c for c in cuts if a0 + 0.3 * scale < c < a1 - 0.3 * scale]
            if not inner:
                out.append(s)
                continue
            gap = 0.5 * (s.thickness + 4)
            bounds = [a0] + [v for c in inner for v in (c - gap, c + gap)] + [a1]
            for lo, hi in zip(bounds[::2], bounds[1::2]):
                if hi - lo >= MIN_PIECE:
                    out.append(Segment(lo, s.y0, hi, s.y1, s.thickness, False, "h") if orient == "h" else
                               Segment(s.x0, lo, s.x1, hi, s.thickness, False, "v"))
        return out

    for s in split(lw.horizontal, xs, "h"):
        if not (x_lo - 1.0 * scale <= s.x0 and s.x1 <= x_hi + 1.0 * scale and y_top - 1.0 * scale <= s.y0 <= y_bot + 1.0 * scale):
            continue
        j, d = nearest(ys, s.y0)
        i = bay_of(xs, s.x0, s.x1)
        if i is None:
            continue
        if s.dashed and d <= 0.6 * scale:
            items.append(Item("brace", level, ("brace", yl[j], i, level), sheet, s))
        elif not s.dashed and d <= tol:
            items.append(Item("girder", level, ("girder", yl[j], i, level), sheet, s,
                              geom=dict(y=gy[j], x0=ax_x(s.x0), x1=ax_x(s.x1), i=i, j=j)))
    for s in split(lw.vertical, ys, "v"):
        if not (y_top - 1.0 * scale <= s.y0 and s.y1 <= y_bot + 1.0 * scale and x_lo - 1.0 * scale <= s.x0 <= x_hi + 1.0 * scale):
            continue
        i, d = nearest(xs, s.x0)
        j = bay_of(ys, s.y0, s.y1)
        if j is None:
            continue
        if s.dashed and d <= 0.6 * scale:
            items.append(Item("brace", level, ("brace", xl[i], j, level), sheet, s))
        elif not s.dashed and d <= tol:
            items.append(Item("beam", level, ("beam", xl[i], j, level), sheet, s,
                              geom=dict(x=gx[i], y0=ax_y(s.y1), y1=ax_y(s.y0), i=i, j=j)))
        elif not s.dashed:
            bi = bay_of(xs, s.x0, s.x0 + 0.01)
            if bi is None:
                warnings.append(f"vertical member at x={ax_x(s.x0):.2f} m outside the grid, ignored")
                continue
            items.append(Item("joist", level, ("joist", bi, j, level, round(ax_x(s.x0), 2)), sheet, s,
                              geom=dict(x=ax_x(s.x0), y0=ax_y(s.y1), y1=ax_y(s.y0), i=bi, j=j)))
    # number joists within each bay left to right, as the mark grammar does
    groups: dict = {}
    for it in items:
        if it.kind == "joist":
            groups.setdefault((it.key[1], it.key[2]), []).append(it)
    for (bi, j), js in groups.items():
        for m, it in enumerate(sorted(js, key=lambda t: t.geom["x"]), 1):
            it.key = ("joist", bi, j, level, m)

    # columns: one per square; their tag sits at the square's lower right
    col_items = []
    for s in lw.squares:
        i, _ = nearest(xs, s.cx)
        j, _ = nearest(ys, s.cy)
        it = Item("column", level, ("column", yl[j], i, level), sheet, None, geom=dict(x=gx[i], y=gy[j], i=i, j=j))
        it.square = s
        col_items.append(it)

    tag_boxes = [b for b in boxes if TAGGY.search(normalise(b.text))]
    rest = []
    for b in tag_boxes:
        if "[" in b.text or b.orientation != "h" or _tag_target(b) != "any":
            rest.append(b)
            continue
        best, best_d = None, 2.2 * sq
        for it in col_items:
            s = it.square
            d = np.hypot(b.bbox[0] - (s.cx + s.size / 2), b.bbox[1] - (s.cy + s.size / 2))
            if b.bbox[0] > s.cx and b.bbox[1] > s.cy and d < best_d:
                best, best_d = it, d
        (best.texts.append(b) if best else rest.append(b))

    def accept(b, it):
        tgt = _tag_target(b)
        if it.kind == "brace":
            ok = tgt in ("brace", "any")
        else:
            ok = tgt in ("member", "any")
        return ok and b.orientation == it.seg.orient
    _assign(rest, items, accept, max_dist=max(30.0, 0.9 * scale))
    return PlanResult(sheet, level, tos, xs, ys, gx, gy, scale, items + col_items, read, inferred, warnings)


def level_annotations(boxes: list[TextBox]) -> dict[int, float]:
    """{level: elevation} from elevation-sheet annotations ("LEVEL n T.O.S." over "EL. +z")."""
    out = {}
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        if not t.startswith("EL"):
            continue
        z = parse_number(t)
        above = [a for a in boxes if a is not b and abs(a.bbox[0] - b.bbox[0]) < 3 * b.height and
                 0.4 * b.height < b.cy - a.cy < 2.2 * b.height]
        if z is None or not above:
            continue
        m = re.search(r"LEVEL\s*(\d+)", normalise(above[0].text))
        if m and z > 0:
            out[int(m.group(1))] = z
    return out


# ----------------------------------------------------------------------------- elevations
@dataclass
class ElevationResult:
    sheet: str
    frame: str
    items: list
    warnings: list


def parse_elevation(sheet: str, frame: str, boxes: list[TextBox], lw: Linework,
                    grid_x: list, grid_y: list, levels: dict) -> ElevationResult:
    warnings = []
    along_x = frame.isalpha()
    st_m = grid_x if along_x else grid_y
    st_lab = [str(i + 1) for i in range(len(grid_x))] if along_x else [LETTERS[j] for j in range(len(grid_y))]

    cols = [s for s in lw.vertical if not s.dashed and s.thickness >= 4.0]
    if not cols:
        raise ParseError(f"{sheet}: no columns found")
    longest = max(s.length for s in cols)
    cols = [s for s in cols if s.length >= 0.4 * longest]
    st_px = cluster([s.x0 for s in cols], tol=6)
    if len(st_px) != len(st_m):
        warnings.append(f"{len(st_px)} column lines but {len(st_m)} gridlines on the plans")
    n = min(len(st_px), len(st_m))
    st_px, st_m, st_lab = st_px[:n], st_m[:n], st_lab[:n]
    scale_x = (st_px[-1] - st_px[0]) / (st_m[-1] - st_m[0]) if n > 1 else None

    # vertical reference from the level annotations ("LEVEL n T.O.S." over "EL. +z")
    pts = []
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        if not t.startswith("EL"):
            continue
        z = parse_number(t)
        if z is None:
            continue
        above = [a for a in boxes if a is not b and abs(a.bbox[0] - b.bbox[0]) < 3 * b.height and
                 0.4 * b.height < b.cy - a.cy < 2.2 * b.height]
        y_line = (b.cy + above[0].cy) / 2 if above else b.cy - 0.55 * b.height
        pts.append((z, y_line))
    if len(pts) >= 2:
        zs, yv = np.array(pts).T
        k, c = np.polyfit(zs, yv, 1)                   # y = k*z + c, k < 0
        scale_z = -k
        if scale_x and abs(scale_z - scale_x) / scale_x > 0.03:
            warnings.append(f"vertical scale {scale_z:.1f} px/m disagrees with horizontal {scale_x:.1f} px/m")
    else:
        warnings.append("level annotations not read; vertical scale taken from the column lines")
        top = max(levels.values())
        y_bot, y_top = max(s.y1 for s in cols), min(s.y0 for s in cols)
        scale_z = (y_bot - y_top) / top
        k, c = -scale_z, y_bot

    def z_of(y):
        return (y - c) / k

    ax_h = Axis(st_px, st_m, scale_x or scale_z)
    lv_list = sorted(levels.items())
    items: list[Item] = []

    def bay_of(px0, px1):
        mid = (px0 + px1) / 2
        for i in range(n - 1):
            if st_px[i] < mid < st_px[i + 1]:
                return i
        return None

    for s in lw.horizontal:
        margin = 0.6 * (scale_x or scale_z)
        if s.dashed or not (st_px[0] - margin <= s.x0 and s.x1 <= st_px[-1] + margin):
            continue
        z = z_of(s.y0)
        if z < 0.3:                                     # base plates sit at the foot of the columns
            i, d = nearest(st_px, (s.x0 + s.x1) / 2)
            if d < 0.6 * scale_z and s.length < 1.2 * scale_z:
                key = ("baseplate", frame, i) if along_x else ("baseplate", st_lab[i], int(frame) - 1)
                it = Item("baseplate", 0, key, sheet, s)
                items.append(it)
            continue
        i = bay_of(s.x0, s.x1)
        if i is None or s.length < 0.3 * (st_px[i + 1] - st_px[i]):
            continue
        k_lv = min(lv_list, key=lambda kv: abs(kv[1] - z - 0.25))[0]
        kind = "girder" if along_x else "beam"
        items.append(Item(kind, k_lv, (kind, frame, i, k_lv), sheet, s, geom=dict(z=z)))

    zs_lv = [0.0] + [z for _, z in lv_list]
    for s in cols:
        i, d = nearest(st_px, s.x0)
        if d > 6:
            continue
        z_lo, z_hi = z_of(s.y1), z_of(s.y0)
        for k_idx, (k_lv, z_top) in enumerate(lv_list):
            z_bot = zs_lv[k_idx]
            if z_lo <= z_bot + 0.3 and z_hi >= z_top - 0.3:
                piece = Segment(s.x0, c + k * z_top, s.x1, c + k * z_bot, s.thickness, False, "v")
                key = ("column", frame, i, k_lv) if along_x else ("column", st_lab[i], int(frame) - 1, k_lv)
                items.append(Item("column", k_lv, key, sheet, piece))

    for s in lw.diagonal:
        i = bay_of(s.x0, s.x1)
        if i is None:
            continue
        za, zb = z_of(s.y0), z_of(s.y1)
        zmid = (za + zb) / 2
        k_idx = next((q for q in range(len(lv_list)) if zs_lv[q] <= zmid <= zs_lv[q + 1]), None)
        if k_idx is None:
            continue
        k_lv = lv_list[k_idx][0]
        items.append(Item("brace", k_lv, ("brace", frame, i, k_lv), sheet, s,
                          geom=dict(a=(ax_h(s.x0), za), b=(ax_h(s.x1), zb))))

    tag_boxes = [b for b in boxes if TAGGY.search(normalise(b.text))]

    def accept(b, it):
        tgt = _tag_target(b)
        if it.kind == "brace":
            return tgt in ("brace",) or (tgt == "any" and b.orientation == "d")
        if it.kind == "column":
            return tgt in ("column", "any") and b.orientation == "v"
        if it.kind == "baseplate":
            return tgt in ("baseplate",)
        return tgt in ("member", "any") and b.orientation == "h"
    left = _assign(tag_boxes, [it for it in items if it.kind != "baseplate"], accept, max_dist=max(30.0, 0.8 * scale_z))
    for b in left:                                      # base plate tags hang below the plate
        if _tag_target(b) != "baseplate":
            continue
        plates = [it for it in items if it.kind == "baseplate"]
        if plates:
            min(plates, key=lambda it: np.hypot(it.seg.mid[0] - b.cx, it.seg.mid[1] - b.cy)).texts.append(b)
    for it in items:
        if it.kind in ("girder", "beam", "brace") and it.geom.get("a") is None and it.seg is not None:
            it.geom.update(h0=ax_h(it.seg.x0), h1=ax_h(it.seg.x1))
    return ElevationResult(sheet, frame, items, warnings)


# ----------------------------------------------------------------------------- notes + schedule
KIND_WORDS = {"COLUMN": "column", "GIRDER": "girder", "BEAM": "beam", "JOIST": "joist",
              "BRACE": "brace", "BASEPLATE": "baseplate", "BASE PLATE": "baseplate"}


def parse_schedule(boxes: list[TextBox]) -> dict:
    head = {}
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        for key, name in (("MEMBERTYPE", "type"), ("SECTION", "section"), ("LEVEL", "level"), ("QTY", "qty")):
            if t == key:
                head[name] = b
    rows = []
    if "type" in head:
        type_boxes = [b for b in boxes if normalise(b.text).replace(" ", "") in KIND_WORDS and b.cy > head["type"].cy
                      and abs(b.cx - head["type"].cx) < 120]
        for tb in type_boxes:
            same_row = [b for b in boxes if abs(b.cy - tb.cy) < 0.6 * tb.height and b is not tb]

            def cell(col):
                if col not in head:
                    return None
                c = [b for b in same_row if abs(b.cx - head[col].cx) < 120]
                return c[0].text if c else None
            kind = KIND_WORDS[normalise(tb.text).replace(" ", "")]
            sec, _ = match_section(cell("section") or "")
            lv_raw = normalise(cell("level") or "").replace(" ", "")
            lv = 0 if lv_raw.startswith("FDN") else int(lv_raw[1:]) if re.fullmatch(r"L\d", lv_raw) else None
            q = cell("qty")
            qty = int(q) if q and q.strip().isdigit() else None
            rows.append(dict(kind=kind, section=sec, level=lv, qty=qty))
    notes = {}
    for b in boxes:
        t = normalise(b.text)
        m = re.search(r"BASE\s*PLATES?\s*(PL[0-9X/ ]+?)\s*AT\s*ALL\s*COLUMNS", t)
        if m:
            notes["baseplate_section"] = match_section(m.group(1))[0]
        m = re.search(r"GRID:?\s*(\d+)\s*NUMBERED.*?(\d+)\s*LETTERED.*?(\d+)\s*LEVEL", t)
        if m:
            notes["grid_counts"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    return dict(rows=rows, notes=notes)
