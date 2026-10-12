"""Merge per-sheet readings into one design graph, reconcile it, and write it out.

    python -m ferrosight.parser.parse data/synth/scene_0000            # writes parsed_design.json
    python -m ferrosight.parser.parse data/synth/scene_0000 --overlays # also writes annotated sheets
"""
from __future__ import annotations

import json
import re
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..datagen import sections as S
from ..datagen.structure import LETTERS
from .ocr import SheetOCR
from .preprocess import damage, prepare
from .raster import extract
from .sheets import ParseError, level_annotations, parse_elevation, parse_plan, parse_schedule, sheet_kind
from .text import edit_distance, normalise, valid_mark


def expected_mark(key: tuple, xl: list, yl: list) -> str:
    kind = key[0]
    if kind == "girder":
        _, L, i, k = key
        return f"G-{L}{xl[i]}{xl[i+1]}-L{k}"
    if kind == "beam":
        _, n, j, k = key
        return f"B-{n}{yl[j]}{yl[j+1]}-L{k}"
    if kind == "joist":
        _, i, j, k, m = key
        return f"J-{xl[i]}{xl[i+1]}.{m}{yl[j]}{yl[j+1]}-L{k}"
    if kind == "brace":
        _, f, b, k = key
        return f"BR-{f}{xl[b]}{xl[b+1]}-L{k}" if f.isalpha() else f"BR-{f}{yl[b]}{yl[b+1]}-L{k}"
    if kind == "column":
        _, L, i, k = key
        return f"C-{L}{xl[i]}-L{k}"
    if kind == "baseplate":
        _, L, i = key
        return f"BP-{L}{xl[i]}"
    raise ValueError(key)


def grid_text(key, xl, yl) -> str:
    kind = key[0]
    if kind == "girder":
        return f"{key[1]}/{xl[key[2]]}-{xl[key[2]+1]}"
    if kind == "beam":
        return f"{key[1]}/{yl[key[2]]}-{yl[key[2]+1]}"
    if kind == "joist":
        return f"{xl[key[1]]}-{xl[key[1]+1]}/{yl[key[2]]}-{yl[key[2]+1]}"
    if kind == "brace":
        f, b = key[1], key[2]
        return f"{f}/{xl[b]}-{xl[b+1]}" if f.isalpha() else f"{f}/{yl[b]}-{yl[b+1]}"
    return f"{key[1]}/{xl[key[2]]}"


def infer_frame(boxes) -> str | None:
    """Gridline an elevation shows, from marks like G-A12 (frame A), B-1AB or BR-1AB (frame 1)."""
    votes = Counter()
    for b in boxes:
        t = normalise(b.text).replace(" ", "")
        for a, n in re.findall(r"(?:BR-|G-)([A-H])\d\d|(?:BR-|(?<![A-Z])B-)(\d)[A-H]{2}", t):
            votes[a or n] += 1
    return votes.most_common(1)[0][0] if votes else None


def _pl_thickness(section: str) -> float:
    m = re.search(r"x(\d+)$", section or "")
    return int(m.group(1)) / 1000 if m else S.BASEPLATE_THICKNESS


def parse_scene(scene_dir: str | Path, ocr: SheetOCR | None = None, overlays: bool = False,
                damage_level: str | None = None, out_name: str = "parsed_design.json") -> dict:
    """Parse one scene's drawing set. `damage_level` ('light' / 'heavy') degrades every sheet
    first, to measure robustness to scanned or photocopied drawings."""
    t0 = time.time()
    scene_dir = Path(scene_dir)
    sheets = sorted((scene_dir / "drawings").glob("S-*.png"))
    if not sheets:
        raise ParseError(f"no sheets in {scene_dir / 'drawings'}")
    ocr = ocr or SheetOCR(tile=4000, overlap=0, scale=1.0)
    read = {}
    for p in sheets:
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if damage_level:
            gray = damage(gray, damage_level, seed=zlib.crc32(f"{scene_dir.name}/{p.stem}".encode()))
        gray, pre = prepare(gray)
        lw = extract(gray)
        boxes = ocr.read(lw.text_image)
        kind, frame = sheet_kind(boxes, p.stem)
        read[p.stem] = dict(gray=gray, lw=lw, boxes=boxes, kind=kind, frame=frame, pre=pre)

    warnings: list[str] = []
    schedule = dict(rows=[], notes={})
    for name, r in read.items():
        if r["kind"] == "schedule":
            schedule = parse_schedule(r["boxes"])
    sched_sections = defaultdict(set)
    for row in schedule["rows"]:
        if row["section"]:
            sched_sections[(row["kind"], row["level"])].add(row["section"])

    plans = []
    for name, r in read.items():
        if r["kind"] == "plan":
            hint = int(name[-1]) if name[-1].isdigit() else None
            plans.append(parse_plan(name, r["boxes"], r["lw"], hint))
            warnings += [f"{name}: {w}" for w in plans[-1].warnings]
    if not plans:
        raise ParseError("no framing plans recognised")
    plans.sort(key=lambda p: p.level)
    base = plans[0]
    grid_x, grid_y = base.grid_x, base.grid_y
    for p in plans[1:]:
        if len(p.grid_x) != len(grid_x) or len(p.grid_y) != len(grid_y) or \
                max(np.abs(np.subtract(p.grid_x, grid_x)).max(), np.abs(np.subtract(p.grid_y, grid_y)).max()) > 0.05:
            warnings.append(f"{p.sheet}: grid differs from {base.sheet}")
    xl = [str(i + 1) for i in range(len(grid_x))]
    yl = [LETTERS[j] for j in range(len(grid_y))]
    from_elev = {}
    for r in read.values():
        if r["kind"] == "elevation":
            from_elev.update(level_annotations(r["boxes"]))
    levels = {}
    for p in plans:
        if p.tos is None:
            p.tos = from_elev.get(p.level)
            warnings.append(f"{p.sheet}: top-of-steel elevation not read on the plan"
                            + (f"; taken from the elevations (+{p.tos:.3f})" if p.tos is not None else ""))
        elif p.level in from_elev and abs(from_elev[p.level] - p.tos) > 0.005:
            warnings.append(f"{p.sheet}: plan says T.O.S. +{p.tos:.3f}, elevations say +{from_elev[p.level]:.3f}")
        levels[p.level] = p.tos
    if any(v is None for v in levels.values()):
        raise ParseError("missing level elevations")

    for name, r in read.items():
        if r["kind"] == "elevation" and not r["frame"]:
            r["frame"] = infer_frame(r["boxes"])
            warnings.append(f"{name}: title not read; frame {r['frame']} inferred from member marks"
                            if r["frame"] else f"{name}: title not read and no marks to go on; sheet skipped")
    elevations = []
    for name, r in read.items():
        if r["kind"] == "elevation" and r["frame"]:
            try:
                elevations.append(parse_elevation(name, r["frame"], r["boxes"], r["lw"], grid_x, grid_y, levels))
                warnings += [f"{name}: {w}" for w in elevations[-1].warnings]
            except ParseError as exc:
                warnings.append(str(exc))

    # ---------------------------------------------------------------- merge readings by location
    found: dict[tuple, list] = defaultdict(list)
    for src in plans + elevations:
        for it in src.items:
            found[it.key].append(it)
    if "baseplate_section" in schedule["notes"] or any(r["kind"] == "baseplate" for r in schedule["rows"]):
        for (kind, L, i, k) in [key for key in found if key[0] == "column" and key[3] == 1]:
            found.setdefault(("baseplate", L, i), [])         # note 3: base plates at all columns

    def section_for(key, its):
        kind = key[0]
        lv = 0 if kind == "baseplate" else key[3]
        votes = Counter()
        listed = sched_sections.get((kind, lv), set())
        for it in its:
            sec, how, m, _raw = it.tag(prefer=listed)
            if sec:
                w = 2 if how == "exact" else 1
                w += 2 if m and edit_distance(m, expected_mark(key, xl, yl)) <= 2 else 0  # tag is clearly this member's
                w += 1 if sec in listed else 0
                votes[sec] += w
        if votes:
            sec, _ = votes.most_common(1)[0]
            how = "drawing" if len(votes) == 1 else "drawing (conflict: " + ", ".join(votes) + ")"
            return sec, how
        cands = sched_sections.get((kind, lv), set())
        if kind == "baseplate" and schedule["notes"].get("baseplate_section"):
            cands = {schedule["notes"]["baseplate_section"]}
        if len(cands) == 1:
            return next(iter(cands)), "schedule"
        return None, "unknown"

    def fill_from_peers(comps):
        """Last resort for an unread section: the section every other member of the same type at the
        same level has (then at any level). Flagged so it is never mistaken for a reading."""
        for c in comps:
            if c["section"]:
                continue
            for same_level in (True, False):
                peers = Counter(o["section"] for o in comps if o["kind"] == c["kind"] and o["section"] and
                                o["section_source"] != "inferred from peers" and (o["level"] == c["level"] or not same_level))
                if len(peers) == 1 or (peers and peers.most_common(1)[0][1] >= 0.9 * sum(peers.values())):
                    c["section"] = peers.most_common(1)[0][0]
                    c["section_source"] = "inferred from peers"
                    c["confidence"] = round(max(0.0, c["confidence"] + 0.2), 2)   # 'unknown' cost 0.4, this costs 0.2
                    break

    col_sec = {}
    for key, its in found.items():
        if key[0] == "column":
            col_sec[key] = section_for(key, its)[0]

    def half_col(L, i, k):
        sec = col_sec.get(("column", L, i, k))
        return S.depth(sec) / 2 if sec else 0.15

    comps = []
    for key, its in sorted(found.items(), key=lambda kv: str(kv[0])):
        kind = key[0]
        section, sec_how = section_for(key, its)
        mark = expected_mark(key, xl, yl)
        ocr_marks = [m for it in its for m in [it.tag()[2]] if m]
        if any(m == mark for m in ocr_marks):
            mark_status = "read"
        elif any(edit_distance(m, mark) <= 3 for m in ocr_marks):
            mark_status = "repaired"
        elif any(valid_mark(m) for m in ocr_marks):
            mark_status = "conflict"
            warnings.append(f"{mark}: tag reads {ocr_marks[0]} but position says {mark}")
        else:
            mark_status = "derived"
        d = S.depth(section) if section else 0.4
        level = 0 if kind == "baseplate" else key[3]
        up, frame = [0, 0, 1], ""
        conf = 1.0
        if kind == "column":
            L, i, k = key[1], key[2], key[3]
            x, y = grid_x[i], grid_y[LETTERS.index(L)]
            z0 = 0.0 if k == 1 else levels[k - 1]
            start, end, up = [x, y, z0], [x, y, levels[k]], [0, 1, 0]
        elif kind == "baseplate":
            L, i = key[1], key[2]
            x, y = grid_x[i], grid_y[LETTERS.index(L)]
            start, end, up = [x, y, -_pl_thickness(section)], [x, y, 0.0], [0, 1, 0]
            if not its:
                conf -= 0.2                                   # placed from the general note only
        elif kind in ("girder", "beam"):
            plan = next((it for it in its if it.sheet.startswith("S-1")), None)
            k = key[3]
            zc = round(levels[k] - d / 2, 4)
            if kind == "girder":
                L, i = key[1], key[2]
                y = grid_y[LETTERS.index(L)]
                a, b = grid_x[i] + half_col(L, i, k), grid_x[i + 1] - half_col(L, i + 1, k)
                start, end, frame = [a, y, zc], [b, y, zc], L
            else:
                n, j = key[1], key[2]
                x = grid_x[int(n) - 1]
                a = grid_y[j] + half_col(yl[j], int(n) - 1, k)
                b = grid_y[j + 1] - half_col(yl[j + 1], int(n) - 1, k)
                start, end, frame = [x, a, zc], [x, b, zc], n
            if plan is None:
                conf -= 0.3
                warnings.append(f"{mark}: shown on elevation only")
        elif kind == "joist":
            plan = its[0]
            k = key[3]
            zc = round(levels[k] - d / 2, 4)
            start = [plan.geom["x"], plan.geom["y0"], zc]
            end = [plan.geom["x"], plan.geom["y1"], zc]
        elif kind == "brace":
            f, b, k = key[1], key[2], key[3]
            frame = f
            elev = next((it for it in its if "a" in it.geom), None)
            fixed = grid_y[LETTERS.index(f)] if f.isalpha() else grid_x[int(f) - 1]
            if elev is not None:
                (ha, za), (hb, zb) = elev.geom["a"], elev.geom["b"]
            else:
                lo = grid_x if f.isalpha() else grid_y
                z_lo = (0.0 if k == 1 else levels[k - 1]) + 0.15
                ha, hb, za, zb = lo[b] + 0.2, lo[b + 1] - 0.2, z_lo, levels[k] - 0.45
                conf -= 0.4
                warnings.append(f"{mark}: no elevation shows this brace; diagonal direction assumed")
            if f.isalpha():
                start, end = [ha, fixed, za], [hb, fixed, zb]
            else:
                start, end = [fixed, ha, za], [fixed, hb, zb]
        if sec_how == "schedule":
            conf -= 0.2
        elif sec_how == "unknown":
            conf -= 0.4
        if mark_status in ("conflict",):
            conf -= 0.2
        comps.append(dict(
            id=mark, kind=kind, section=section, start=[round(v, 4) for v in start], end=[round(v, 4) for v in end],
            up=up, level=level, grid=grid_text(key, xl, yl) if kind != "column" and kind != "baseplate" else f"{key[1]}/{xl[key[2]]}",
            frame=frame, confidence=round(max(0.0, conf), 2), section_source=sec_how, mark_status=mark_status,
            mark_read=ocr_marks[0] if ocr_marks else None, evidence=[it.evidence() for it in its]))

    fill_from_peers(comps)
    for c in comps:                       # vertical position of beams depends on the (now known) section
        if c["kind"] in ("girder", "beam", "joist") and c["section_source"] == "inferred from peers":
            zc = round(levels[c["level"]] - S.depth(c["section"]) / 2, 4)
            c["start"][2] = c["end"][2] = zc

    # ---------------------------------------------------------------- reconcile against the schedule
    checks = []
    counts = Counter((c["kind"], c["section"], c["level"]) for c in comps)
    for row in schedule["rows"]:
        if row["qty"] is None:
            continue
        got = counts.get((row["kind"], row["section"], row["level"]), 0)
        checks.append(dict(kind=row["kind"], section=row["section"], level=row["level"], schedule=row["qty"],
                           parsed=got, ok=got == row["qty"]))
        if got != row["qty"]:
            warnings.append(f"schedule lists {row['qty']} {row['kind']} {row['section']} at level {row['level']}, "
                            f"drawings show {got}")
    gc = schedule["notes"].get("grid_counts")
    if gc and gc != [len(grid_x), len(grid_y), len(levels)]:
        warnings.append(f"general note says grid {gc}, plans show {[len(grid_x), len(grid_y), len(levels)]}")

    out = dict(name=scene_dir.name, grid_x=grid_x, grid_y=grid_y, x_labels=xl, y_labels=yl,
               levels=[levels[k] for k in sorted(levels)], components=comps,
               sheets={n: dict(kind=r["kind"], frame=r["frame"], text_boxes=len(r["boxes"]), **r["pre"]) for n, r in read.items()},
               schedule=schedule, schedule_checks=checks, warnings=warnings,
               stats=dict(seconds=round(time.time() - t0, 1), components=len(comps),
                          dims_read=sum(p.dims_read for p in plans), dims_inferred=sum(p.dims_inferred for p in plans)))
    (scene_dir / out_name).write_text(json.dumps(out, indent=1))
    if overlays:
        write_overlays(scene_dir, read, comps)
    return out


COLORS = dict(column=(200, 0, 200), girder=(0, 140, 0), beam=(0, 110, 220), joist=(0, 170, 170),
              brace=(0, 0, 230), baseplate=(140, 70, 0))


def write_overlays(scene_dir: Path, read: dict, comps: list) -> None:
    out_dir = scene_dir / "parsed_overlays"
    out_dir.mkdir(exist_ok=True)
    for name, r in read.items():
        vis = cv2.cvtColor(r["gray"], cv2.COLOR_GRAY2BGR)
        vis = cv2.addWeighted(vis, 0.45, np.full_like(vis, 255), 0.55, 0)
        for c in comps:
            for ev in c["evidence"]:
                if ev["sheet"] != name:
                    continue
                col = COLORS[c["kind"]] if c["confidence"] >= 0.8 else (0, 120, 255)
                if "line_px" in ev:
                    x0, y0, x1, y1 = map(int, ev["line_px"])
                    cv2.line(vis, (x0, y0), (x1, y1), col, 4)
                if "tag_bbox_px" in ev:
                    x0, y0, x1, y1 = map(int, ev["tag_bbox_px"])
                    cv2.rectangle(vis, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), col, 2)
        cv2.imwrite(str(out_dir / f"{name}.png"), vis)
