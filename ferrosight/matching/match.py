"""Phase 4: match each member of the parsed design to what the footage shows.

For every member and analysed frame, sample points on the member are projected into the image
with the camera pose. SAM 3's masks are 2D, so steel seen at a pixel could belong to any member
along that ray. A point is therefore used as evidence only where no OTHER design member projects
(in front or behind): there, steel can only mean this member is present. Across frames:

  present       steel where the member should be, in most clear views
  absent        clear views of the spot and no steel there (missing, or moved elsewhere)
  inconclusive  clear views disagree; left for the verifier (phase 5)
  not_visible   fewer than MIN_VIEWS clear views

    python -m ferrosight.matching.match data/synth/scene_0000      # one scene
    python -m ferrosight.matching.match data/synth --evaluate      # all scenes, scored against labels
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..datagen.sections import SECTIONS

N_POINTS = 15
MIN_VISIBLE = 0.5      # share of sample points in view and not hidden by other members
MIN_CLEAN = 3          # points with no other member on the same pixels, needed for a view to count
MIN_SIZE_PX = 8        # projected size at mask resolution for a view to count
MIN_VIEWS = 3
HIT = 0.4              # share of clean points on steel for a view to say "steel here"
PRESENT = 0.5          # share of clear views saying "steel here" to call it present
ABSENT = 0.15          # at most this share of clear views with steel -> absent
KERNEL = np.ones((3, 3), np.uint8)


def project(points, cam_to_world, K):
    w2c = np.linalg.inv(np.asarray(cam_to_world, float))
    pc = points @ w2c[:3, :3].T + w2c[:3, 3]
    z = -pc[:, 2]                                   # distance in front (camera looks down -Z)
    zz = np.where(z > 1e-6, z, 1e-6)
    u = K["fx"] * pc[:, 0] / zz + K["cx"]
    v = K["cy"] - K["fy"] * pc[:, 1] / zz
    return u, v, z


def sample_points(c) -> np.ndarray:
    a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
    if c["kind"] == "baseplate":                   # a plate is an area, not a line
        g = np.linspace(-0.2, 0.2, 4)
        x, y = np.meshgrid(g, g)
        mid = (a + b) / 2
        return np.c_[mid[0] + x.ravel(), mid[1] + y.ravel(), np.full(x.size, mid[2])]
    return a + np.linspace(0.08, 0.92, N_POINTS)[:, None] * (b - a)


def width_of(c) -> float:
    s = SECTIONS.get(c.get("section") or "")
    return max(s["d"], s["bf"]) if s else 0.3


def design_buffers(comps, c2w, K, sc, h, w):
    """index: member nearest the camera in each mask cell (painter's algorithm).
    cover: how many members' footprints (grown by one cell) reach each cell."""
    index = np.full((h, w), -1, np.int32)
    cover = np.zeros((h, w), np.int32)
    order = []
    for i, c in enumerate(comps):
        if c["kind"] == "baseplate":
            mid = (np.asarray(c["start"], float) + np.asarray(c["end"], float)) / 2
            pts = mid + np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0], [-.25, .25, 0]])
        else:
            a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
            pts = a + np.linspace(0, 1, 9)[:, None] * (b - a)
        u, v, z = project(pts, c2w, K)
        front = z > 0.1
        if front.sum() < 2 or (c["kind"] == "baseplate" and not front.all()):
            continue
        order.append((float(z[front].mean()), i, np.clip(u[front] / sc, -2e4, 2e4), np.clip(v[front] / sc, -2e4, 2e4)))
    for zm, i, u, v in sorted(order, key=lambda t: -t[0]):           # far first, near last
        fp = np.zeros((h, w), np.uint8)
        pts = np.round(np.c_[u, v]).astype(np.int32).reshape(-1, 1, 2)
        if comps[i]["kind"] == "baseplate":
            cv2.fillConvexPoly(fp, pts, 1)
        else:
            t = int(np.clip(round(K["fx"] / sc * width_of(comps[i]) / max(zm, 0.1)), 1, 40))
            cv2.polylines(fp, [pts], False, 1, thickness=t)
        index[fp > 0] = i
        cover += cv2.dilate(fp, KERNEL)
    return index, cover


def match_scene(scene: Path) -> dict:
    design = json.loads((scene / "parsed_design.json").read_text())
    cam = json.loads((scene / "render" / "cameras.json").read_text())
    meta = json.loads((scene / "perception" / "sam3.json").read_text())
    p = np.load(scene / "perception" / "sam3.npz")
    sc = cam["mask_scale"]
    h, w = meta["mask_shape"]
    keep = p["score"] >= meta["threshold"]
    comps = design["components"]
    pts_of = [sample_points(c) for c in comps]
    views = defaultdict(list)
    for f in meta["frames"]:
        sel = keep & (p["frame"] == f)
        steel = p["masks"][sel].any(0) if sel.any() else np.zeros((h, w), bool)
        steel = cv2.dilate(steel.astype(np.uint8), KERNEL) > 0     # one cell of slack for thin members
        c2w = cam["frames"][f]["cam_to_world"]
        index, cover = design_buffers(comps, c2w, cam, sc, h, w)
        for i, c in enumerate(comps):
            u, v, z = project(pts_of[i], c2w, cam)
            mu, mv = np.floor(u / sc).astype(int), np.floor(v / sc).astype(int)
            inside = (z > 0.1) & (mu >= 0) & (mu < w) & (mv >= 0) & (mv < h)
            if inside.sum() < 2:
                continue
            vis = np.zeros(len(u), bool)
            vis[inside] = index[mv[inside], mu[inside]] == i
            if vis.mean() < MIN_VISIBLE:
                continue
            if np.hypot(np.ptp(u[inside]), np.ptp(v[inside])) / sc < MIN_SIZE_PX:
                continue
            clean = vis.copy()
            clean[vis] = cover[mv[vis], mu[vis]] == 1                  # only this member could be here
            if clean.sum() < MIN_CLEAN:
                continue
            hit = float(steel[mv[clean], mu[clean]].mean())
            views[c["id"]].append(dict(frame=int(f), visible=round(float(vis.mean()), 2),
                                       clean=int(clean.sum()), steel=round(hit, 2),
                                       bbox_px=[int(u[inside].min()), int(v[inside].min()),
                                                int(u[inside].max()), int(v[inside].max())]))
    members = []
    for c in comps:
        vs = views.get(c["id"], [])
        if len(vs) < MIN_VIEWS:
            verdict, ratio = "not_visible", None
        else:
            ratio = float(np.mean([x["steel"] >= HIT for x in vs]))
            verdict = "present" if ratio >= PRESENT else "absent" if ratio <= ABSENT else "inconclusive"
        key = (lambda x: (-x["steel"], -x["clean"])) if verdict == "present" else (lambda x: (x["steel"], -x["clean"]))
        members.append(dict(id=c["id"], kind=c["kind"], section=c["section"], grid=c["grid"], level=c["level"],
                            verdict=verdict, clear_views=len(vs),
                            steel_ratio=None if ratio is None else round(ratio, 2),
                            evidence=sorted(vs, key=key)[:4]))
    out = dict(scene=scene.name, frames=meta["frames"], members=members,
               summary=dict(Counter(m["verdict"] for m in members)))
    (scene / "matching.json").write_text(json.dumps(out, indent=1))
    return out


EXPECTED = dict(ok="present", wrong_section="present", missing="absent", misplaced="absent", not_visible="not_visible")
PRED = ["present", "absent", "inconclusive", "not_visible"]


def evaluate(scenes: list[Path]) -> str:
    conf = defaultdict(Counter)
    defects = defaultdict(Counter)
    for s in scenes:
        labels = json.loads((s / "labels.json").read_text())["components"]
        res = json.loads((s / "matching.json").read_text())
        for m in res["members"]:
            lab = labels.get(m["id"])
            if lab is None:
                continue
            conf[EXPECTED[lab["expected_verdict"]]][m["verdict"]] += 1
            if lab["status"] != "ok":
                defects[(lab["status"], "seen" if lab["observable"] else "not seen")][m["verdict"]] += 1
    lines = [f"Matching on {len(scenes)} scenes", "",
             "| Truth \\ Verdict | " + " | ".join(PRED) + " |", "| --- |" + " ---: |" * len(PRED)]
    for exp in ["present", "absent", "not_visible"]:
        lines.append(f"| {exp} | " + " | ".join(str(conf[exp][p]) for p in PRED) + " |")
    tp = conf["absent"]["absent"]
    flagged = sum(conf[e]["absent"] for e in conf)
    lines += ["",
              f"Absent alerts: {flagged}. Correct: {tp} (precision {tp / max(1, flagged):.1%}). "
              f"Real absences caught: {tp}/{sum(conf['absent'].values())} "
              f"(recall {tp / max(1, sum(conf['absent'].values())):.1%}). "
              f"False alarms on present members: {conf['present']['absent']}/{sum(conf['present'].values())} "
              f"({conf['present']['absent'] / max(1, sum(conf['present'].values())):.2%}).",
              "", "| Injected defect | Camera saw it | " + " | ".join(PRED) + " |",
              "| --- | --- |" + " ---: |" * len(PRED)]
    for (status, seen), c in sorted(defects.items()):
        lines.append(f"| {status} | {seen} | " + " | ".join(str(c[p]) for p in PRED) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Match parsed design members to SAM 3 observations")
    ap.add_argument("path", type=Path, help="a scene folder, or the data folder for all scenes")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()
    ready = lambda s: (s / "parsed_design.json").exists() and (s / "perception" / "sam3.npz").exists()
    scenes = [args.path] if (args.path / "render").exists() else sorted(s for s in args.path.glob("scene_*") if ready(s))
    for s in scenes:
        r = match_scene(s)
        print(f"{s.name}: {r['summary']}")
    if args.evaluate:
        report = evaluate(scenes)
        print("\n" + report)
        (scenes[0].parent / "matching_eval.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
