"""Render the design as a 2D drawing set (framing plans, elevations, schedule).

Every sheet is saved as PNG plus one combined PDF. `annotations.json` records where
each member and its tag sit on the sheet in pixels: ground truth for the drawing parser.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle

from .structure import Design

SHEET_IN = (17, 11)
DPI = 150
INK = "#111111"


def _sheet(design: Design, number: str, title: str):
    fig = plt.figure(figsize=SHEET_IN, dpi=DPI)
    frame = fig.add_axes([0, 0, 1, 1])
    frame.set_xlim(0, 1), frame.set_ylim(0, 1), frame.axis("off")
    frame.add_patch(Rectangle((0.015, 0.02), 0.97, 0.96, fill=False, lw=1.6, ec=INK))
    frame.add_patch(Rectangle((0.665, 0.02), 0.32, 0.115, fill=False, lw=1.2, ec=INK))
    frame.plot([0.665, 0.985], [0.085, 0.085], color=INK, lw=0.8)
    frame.plot([0.905, 0.905], [0.02, 0.085], color=INK, lw=0.8)
    frame.text(0.672, 0.110, f"PROJECT  {design.name}  STEEL FRAME", fontsize=9, weight="bold", va="center")
    frame.text(0.672, 0.063, title, fontsize=9, weight="bold", va="center")
    frame.text(0.672, 0.036, "SCALE: NTS    UNITS: METRES    REV: 0", fontsize=7, va="center")
    frame.text(0.945, 0.0525, number, fontsize=16, weight="bold", ha="center", va="center")
    ax = fig.add_axes([0.07, 0.17, 0.86, 0.77])
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _bubble(ax, x, y, label, r):
    ax.add_patch(Circle((x, y), r, fill=False, lw=1.0, ec=INK, clip_on=False))
    ax.text(x, y, label, ha="center", va="center", fontsize=9, weight="bold")


def _collect(fig, ax, sheet, drawn, out):
    """Convert drawn members to pixel-space annotations (origin top-left)."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    H = fig.bbox.height
    for comp, (p0, p1), text in drawn:
        a, b = ax.transData.transform(p0), ax.transData.transform(p1)
        ann = dict(sheet=sheet, component_id=comp.id, kind=comp.kind, section=comp.section,
                   line_px=[round(a[0], 1), round(H - a[1], 1), round(b[0], 1), round(H - b[1], 1)])
        if text is not None:
            bb = text.get_window_extent(rend)
            ann["tag_text"] = text.get_text()
            ann["tag_bbox_px"] = [round(bb.x0, 1), round(H - bb.y1, 1), round(bb.x1, 1), round(H - bb.y0, 1)]
        out.append(ann)


def _plan(design: Design, level: int):
    gx, gy = design.grid_x, design.grid_y
    span = max(gx[-1], gy[-1])
    ext, r = 0.14 * span + 1.2, 0.035 * span + 0.35
    z = design.levels[level - 1]
    fig, ax = _sheet(design, f"S-10{level}", f"LEVEL {level} FRAMING PLAN   T.O.S. EL. +{z:.3f}")
    for i, x in enumerate(gx):
        ax.plot([x, x], [-ext, gy[-1] + ext], color="#666", lw=0.6, ls=(0, (12, 3, 2, 3)))
        _bubble(ax, x, gy[-1] + ext + r, design.x_labels[i], r)
    for j, y in enumerate(gy):
        ax.plot([-ext, gx[-1] + ext], [y, y], color="#666", lw=0.6, ls=(0, (12, 3, 2, 3)))
        _bubble(ax, -ext - r, y, design.y_labels[j], r)
    for i in range(len(gx) - 1):      # bay dimensions
        yd = gy[-1] + ext * 0.55
        ax.annotate("", (gx[i], yd), (gx[i + 1], yd), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
        ax.text((gx[i] + gx[i + 1]) / 2, yd + 0.12, f"{gx[i+1]-gx[i]:.3f}", ha="center", va="bottom", fontsize=7)
    for j in range(len(gy) - 1):
        xd = -ext * 0.55
        ax.annotate("", (xd, gy[j]), (xd, gy[j + 1]), arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK))
        ax.text(xd - 0.12, (gy[j] + gy[j + 1]) / 2, f"{gy[j+1]-gy[j]:.3f}", ha="right", va="center", fontsize=7, rotation=90)

    drawn = []
    fs = 6.5 if len(design.components) < 140 else 5.5
    for c in design.components:
        if c.level != level:
            continue
        (x0, y0, _), (x1, y1, _) = c.start, c.end
        if c.kind == "column":
            s = 0.22
            ax.add_patch(Rectangle((x0 - s, y0 - s), 2 * s, 2 * s, fc=INK, ec=INK, zorder=5))
            t = ax.text(x0 + s + 0.08, y0 - s - 0.1, c.section, fontsize=fs - 1, ha="left", va="top", color="#333")
            drawn.append((c, ((x0 - s, y0 - s), (x0 + s, y0 + s)), t))
        elif c.kind in ("girder", "beam", "joist"):
            lw = {"girder": 2.6, "beam": 2.2, "joist": 1.3}[c.kind]
            ax.plot([x0, x1], [y0, y1], color=INK, lw=lw, solid_capstyle="butt", zorder=3)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(x1 - x0) > abs(y1 - y0):
                below = abs(my - gy[-1]) < 1e-6           # top gridline: tag goes inside
                t = ax.text(mx, my + (-0.14 if below else 0.14), f"{c.section}  [{c.id}]", fontsize=fs,
                            ha="center", va="top" if below else "bottom")
            else:
                right = abs(mx - gx[0]) < 1e-6            # first gridline: tag goes inside
                t = ax.text(mx + (0.14 if right else -0.14), my, f"{c.section}  [{c.id}]", fontsize=fs,
                            ha="left" if right else "right", va="center", rotation=90)
            drawn.append((c, ((x0, y0), (x1, y1)), t))
        elif c.kind == "brace":
            off = 0.32
            along_x = abs(x1 - x0) > abs(y1 - y0)
            if along_x:
                sgn = 1 if abs(y0 - gy[-1]) < 1e-6 else -1          # offset away from the building
                p0, p1 = (x0, y0 + sgn * off), (x1, y1 + sgn * off)
                t = ax.text((p0[0] + p1[0]) / 2, p0[1] + sgn * 0.12, f"VB  {c.section}  [{c.id}]", fontsize=fs,
                            ha="center", va="bottom" if sgn > 0 else "top")
            else:
                sgn = 1 if abs(x0 - gx[-1]) < 1e-6 else -1
                p0, p1 = (x0 + sgn * off, y0), (x1 + sgn * off, y1)
                t = ax.text(p0[0] + sgn * 0.12, (p0[1] + p1[1]) / 2, f"VB  {c.section}  [{c.id}]", fontsize=fs,
                            ha="left" if sgn > 0 else "right", va="center", rotation=90)
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.4, ls=(0, (4, 2)), zorder=4)
            drawn.append((c, (p0, p1), t))
    ax.set_xlim(-ext - 2.4 * r, gx[-1] + ext + r)
    ax.set_ylim(-ext - r, gy[-1] + ext + 2.4 * r)
    ax.text(gx[-1] / 2, -ext - 0.4 * r, "VB = VERTICAL BRACING BELOW THIS LEVEL (SEE ELEVATIONS).  "
            "MEMBER MARKS IN [BRACKETS].", fontsize=7, ha="center", va="top", style="italic")
    return fig, ax, drawn


def _elevation(design: Design, frame: str, sheet_no: str):
    along_x = frame in design.y_labels
    stations = design.grid_x if along_x else design.grid_y
    labels = design.x_labels if along_x else design.y_labels
    h = 0 if along_x else 1                      # horizontal coordinate index
    top = design.levels[-1]
    span = max(stations[-1], top)
    r = 0.03 * span + 0.3
    fig, ax = _sheet(design, sheet_no, f"FRAME ELEVATION ON GRIDLINE {frame}")
    for i, s in enumerate(stations):
        ax.plot([s, s], [-1.0, top + 1.2], color="#666", lw=0.6, ls=(0, (12, 3, 2, 3)))
        _bubble(ax, s, top + 1.2 + r, labels[i], r)
    for k, z in enumerate([0.0] + design.levels):
        ax.plot([-1.5, stations[-1] + 1.5], [z, z], color="#999", lw=0.5, ls=":")
        name = "T.O. BASE PLATE" if k == 0 else f"LEVEL {k} T.O.S."
        ax.text(stations[-1] + 1.7, z, f"{name}\nEL. +{z:.3f}", fontsize=7, va="center", ha="left")
    drawn = []
    for c in design.components:
        on_frame = c.frame == frame or (c.kind in ("column", "baseplate") and frame in c.grid.split("/"))
        if not on_frame or c.kind == "joist":
            continue
        p0, p1 = (c.start[h], c.start[2]), (c.end[h], c.end[2])
        if c.kind == "column":
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=3.2, solid_capstyle="butt")
            t = ax.text(p0[0] - 0.14, (p0[1] + p1[1]) / 2, f"{c.section}  [{c.id}]", fontsize=6.5, ha="right", va="center", rotation=90)
        elif c.kind == "baseplate":
            ax.add_patch(Rectangle((p0[0] - 0.25, -0.12), 0.5, 0.12, fc=INK, ec=INK))
            t = ax.text(p0[0], -0.2, f"{c.section} [{c.id}]", fontsize=5.5, ha="center", va="top")
            p0, p1 = (p0[0] - 0.25, -0.12), (p0[0] + 0.25, 0.0)
        elif c.kind == "brace":
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.0)
            ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            ang = ang if -90 <= ang <= 90 else ang - 180
            t = ax.text((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 0.16, f"{c.section}  [{c.id}]", fontsize=6.5,
                        ha="center", va="bottom", rotation=ang, rotation_mode="anchor")
        else:                                       # girder or beam seen in elevation
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=INK, lw=2.6, solid_capstyle="butt")
            t = ax.text((p0[0] + p1[0]) / 2, p0[1] + 0.3, f"{c.section}  [{c.id}]", fontsize=6.5, ha="center", va="bottom")
        drawn.append((c, (p0, p1), t))
    ax.set_xlim(-2.5, stations[-1] + 5.5)
    ax.set_ylim(-1.6, top + 1.2 + 2.4 * r)
    return fig, ax, drawn


def _schedule(design: Design):
    fig, ax = _sheet(design, "S-001", "GENERAL NOTES AND MEMBER SCHEDULE")
    ax.set_aspect("auto"), ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    counts = Counter((c.kind, c.section, c.level) for c in design.components)
    rows = [[k.upper(), s, "FDN" if lv == 0 else f"L{lv}", str(n)]
            for (k, s, lv), n in sorted(counts.items(), key=lambda t: (t[0][2], t[0][0]))]
    tbl = ax.table(cellText=rows, colLabels=["MEMBER TYPE", "SECTION", "LEVEL", "QTY"],
                   bbox=[0.0, max(0.05, 0.95 - 0.045 * (len(rows) + 1)), 0.5, min(0.9, 0.045 * (len(rows) + 1))],
                   cellLoc="center")
    tbl.auto_set_font_size(False), tbl.set_fontsize(8)
    notes = [
        "GENERAL NOTES",
        "1. ALL DIMENSIONS IN METRES UNLESS NOTED OTHERWISE.",
        "2. T.O.S. = TOP OF STEEL. ELEVATIONS REFER TO T.O. BASE PLATE = +0.000.",
        f"3. BASE PLATES {design.components[0].section} AT ALL COLUMNS U.N.O.",
        "4. VERTICAL BRACING (VB) TO BE INSTALLED BEFORE THE NEXT LEVEL IS ERECTED.",
        "5. MEMBER MARKS: C=COLUMN  G=GIRDER  B=BEAM  J=JOIST  BR=BRACE  BP=BASE PLATE.",
        "6. NO MEMBER SHALL BE SUBSTITUTED WITHOUT WRITTEN APPROVAL OF THE ENGINEER.",
        f"7. GRID: {len(design.grid_x)} NUMBERED LINES x {len(design.grid_y)} LETTERED LINES, {len(design.levels)} LEVEL(S).",
    ]
    for n, line in enumerate(notes):
        ax.text(0.56, 0.95 - 0.05 * n, line, fontsize=9, weight="bold" if n == 0 else "normal", va="top")
    return fig


def render_drawings(design: Design, out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    (out / "drawings").mkdir(parents=True, exist_ok=True)
    annotations, sheets = [], []
    jobs = [("S-001", _schedule(design), None, None)]
    for lv in range(1, len(design.levels) + 1):
        fig, ax, drawn = _plan(design, lv)
        jobs.append((f"S-10{lv}", fig, ax, drawn))
    frames = sorted({c.frame for c in design.components if c.kind == "brace"})
    for n, fr in enumerate(frames, 1):
        fig, ax, drawn = _elevation(design, fr, f"S-20{n}")
        jobs.append((f"S-20{n}", fig, ax, drawn))
    with PdfPages(out / "drawings.pdf") as pdf:
        for sheet, fig, ax, drawn in jobs:
            if drawn is not None:
                _collect(fig, ax, sheet, drawn, annotations)
            fig.savefig(out / "drawings" / f"{sheet}.png", dpi=DPI)
            pdf.savefig(fig)
            plt.close(fig)
            sheets.append(sheet)
    meta = dict(sheet_px=[int(SHEET_IN[0] * DPI), int(SHEET_IN[1] * DPI)], sheets=sheets, annotations=annotations)
    (out / "drawings" / "annotations.json").write_text(json.dumps(meta, indent=1))
    return sheets
