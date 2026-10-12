"""Turn render ground truth into benchmark labels.

For every design component we answer two separate questions:
  1. what is its true status on site?   ok | missing | wrong_section | misplaced
  2. could the camera actually tell?    observable = its design location was in view and unoccluded

Question 2 is what lets the benchmark score "not visible" as a correct answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MIN_PIXELS = 60        # full-resolution pixels for a component to count as seen in a frame
MIN_FRAMES = 3         # frames needed before a component / location counts as observable
OCCLUSION_TOL = 0.45   # metres of slack between a sample point and the first surface on its ray


def _project(points, cam_to_world, cam):
    """World points -> (u, v, distance, in_front) using Blender's camera convention (-Z forward)."""
    w2c = np.linalg.inv(np.asarray(cam_to_world))
    pc = (w2c @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    front = pc[:, 2] < -0.05
    z = np.where(front, -pc[:, 2], 1.0)
    u = cam["fx"] * pc[:, 0] / z + cam["cx"]
    v = cam["cy"] - cam["fy"] * pc[:, 1] / z
    return u, v, np.linalg.norm(pc, axis=1), front


def build_labels(scene_dir: str | Path) -> dict:
    scene_dir = Path(scene_dir)
    design = json.loads((scene_dir / "design.json").read_text())
    defects = json.loads((scene_dir / "defects.json").read_text())
    cam = json.loads((scene_dir / "render" / "cameras.json").read_text())
    gt = np.load(scene_dir / "render" / "gt.npz")
    masks, depth = gt["masks"], gt["depth"].astype(np.float32)
    n_frames, mh, mw = masks.shape
    sc = cam["mask_scale"]
    index_of = {cid: i for i, cid in enumerate(cam["id_map"])}

    # direct visibility of everything that physically exists
    seen: dict[str, list] = {cid: [] for cid in cam["id_map"][1:]}
    for f in range(n_frames):
        ids, counts = np.unique(masks[f], return_counts=True)
        for i, n in zip(ids, counts):
            if i == 0 or n * sc * sc < MIN_PIXELS:
                continue
            ys, xs = np.nonzero(masks[f] == i)
            box = [int(xs.min() * sc), int(ys.min() * sc), int((xs.max() + 1) * sc), int((ys.max() + 1) * sc)]
            seen[cam["id_map"][i]].append(dict(frame=f, pixels=int(n * sc * sc), bbox=box))

    def location_views(start, end, own_index):
        """Frames in which the member's design location is in view with nothing in front of it."""
        a, b = np.asarray(start, float), np.asarray(end, float)
        pts = a + np.linspace(0.1, 0.9, 9)[:, None] * (b - a)
        views = []
        for f in range(n_frames):
            u, v, dist, front = _project(pts, cam["frames"][f]["cam_to_world"], cam)
            mu, mv = np.floor(u / sc).astype(int), np.floor(v / sc).astype(int)
            inside = front & (mu >= 0) & (mu < mw) & (mv >= 0) & (mv < mh)
            ok = np.zeros(len(pts), bool)
            if inside.any():
                d = depth[f, mv[inside], mu[inside]]
                own = masks[f, mv[inside], mu[inside]] == own_index if own_index else False
                ok[inside] = (d >= dist[inside] - OCCLUSION_TOL) | own
            frac = float(ok.mean())
            if frac >= 0.5:
                span = float(np.hypot(u[ok].max() - u[ok].min(), v[ok].max() - v[ok].min()))
                views.append(dict(frame=f, fraction=round(frac, 2), span_px=round(span, 1)))
        return [x for x in views if x["span_px"] >= 12 or x["fraction"] == 1.0]

    by_component = {d["component_id"]: d for d in defects}
    components = {}
    for c in design["components"]:
        d = by_component.get(c["id"])
        status = d["type"] if d else "ok"
        direct = sorted(seen.get(c["id"], []), key=lambda s: -s["pixels"])
        loc = sorted(location_views(c["start"], c["end"], index_of.get(c["id"])), key=lambda s: -s["span_px"])
        if status in ("ok", "wrong_section"):
            observable = len(direct) >= MIN_FRAMES
        elif status == "missing":
            observable = len(loc) >= MIN_FRAMES
        else:  # misplaced: the member is seen where it should not be
            observable = len(direct) >= MIN_FRAMES
        components[c["id"]] = dict(
            kind=c["kind"], section=c["section"], level=c["level"], grid=c["grid"], status=status,
            observable=bool(observable), expected_verdict=(status if observable else "not_visible"),
            frames_seen=len(direct), frames_location_in_view=len(loc),
            evidence=direct[:5], location_evidence=loc[:5],
            defect_id=d["defect_id"] if d else None)
    extras = []
    for d in defects:
        if d["type"] == "extra":
            direct = sorted(seen.get(d["component_id"], []), key=lambda s: -s["pixels"])
            extras.append(dict(defect_id=d["defect_id"], component_id=d["component_id"], description=d["description"],
                               observable=len(direct) >= MIN_FRAMES, frames_seen=len(direct), evidence=direct[:5]))
    verdicts = [c["expected_verdict"] for c in components.values()]
    labels = dict(scene=design["name"], n_frames=n_frames, components=components, extras=extras,
                  summary={v: verdicts.count(v) for v in sorted(set(verdicts))} |
                          {"extra_observable": sum(e["observable"] for e in extras), "extra_total": len(extras)})
    (scene_dir / "labels.json").write_text(json.dumps(labels, indent=1))
    return labels


def write_video(scene_dir: str | Path, fps: int = 12) -> bool:
    scene_dir = Path(scene_dir)
    frames = sorted((scene_dir / "render" / "frames").glob("frame_*.jpg"))
    if not frames:
        return False
    try:
        import imageio.v2 as imageio
        with imageio.get_writer(scene_dir / "site_walkthrough.mp4", fps=fps, codec="libx264", quality=8,
                                macro_block_size=1) as w:
            for fr in frames:
                w.append_data(imageio.imread(fr))
        return True
    except Exception as exc:  # video is a convenience; frames are the ground truth
        print(f"[ferrosight] could not write mp4 ({exc}); frames are still in render/frames")
        return False
