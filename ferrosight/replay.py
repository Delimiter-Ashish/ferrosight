"""Inspection replay: the site walkthrough with FerroSight's verdicts painted on.

    python -m ferrosight.replay data/synth/scene_0003 --out docs

Writes docs/replay_<scene>.mp4 (full walkthrough) and docs/replay_<scene>.gif (short loop for the README).
Confirmed members are thin green lines; flagged members are drawn thick with their mark.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from PIL import Image

from .matching.match import project, width_of
from .verify.verify import outcome

BGR = {"flag: absent": (48, 48, 230), "flag: wrong size": (0, 140, 255), "review": (0, 200, 240), "passed": (90, 200, 70)}
LABEL = {"flag: absent": "ABSENT", "flag: wrong size": "WRONG SIZE", "review": "REVIEW"}


def draw_member(img, c, color, cam, f, thick, label=None):
    a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
    if c["kind"] == "baseplate":
        mid = (a + b) / 2
        pts = mid + np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0], [-.25, .25, 0], [-.25, -.25, 0]])
    else:
        pts = a + np.linspace(0, 1, 24)[:, None] * (b - a)
    u, v, z = project(pts, cam["frames"][f]["cam_to_world"], cam)
    front = z > 0.2
    if front.sum() < 2:
        return
    P = np.round(np.c_[np.clip(u[front], -4e3, 5e3), np.clip(v[front], -4e3, 5e3)]).astype(np.int32).reshape(-1, 1, 2)
    if thick:
        t = int(np.clip(cam["fx"] * width_of(c) / max(float(z[front].mean()), 0.2), 4, 60))
        over = img.copy()
        cv2.polylines(over, [P], False, color, t, cv2.LINE_AA)
        cv2.addWeighted(over, 0.5, img, 0.5, 0, dst=img)
        cv2.polylines(img, [P], False, color, 2, cv2.LINE_AA)
        h, w = img.shape[:2]
        cu, cv_ = float(np.median(u[front])), float(np.median(v[front]))
        if label and 0 <= cu < w and 0 <= cv_ < h:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x, y = int(min(max(cu - tw / 2, 4), w - tw - 8)), int(min(max(cv_ - t - 12, 18), h - 8))
            cv2.rectangle(img, (x - 4, y - th - 6), (x + tw + 4, y + 4), color, -1)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.polylines(img, [P], False, color, 1, cv2.LINE_AA)


def header(img, scene, counts):
    bar = np.full((44, img.shape[1], 3), (31, 29, 28), np.uint8)
    cv2.putText(bar, f"FerroSight  |  {scene.replace('scene_', 'building ')}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    x = img.shape[1] - 12
    for key, name in (("passed", "confirmed"), ("review", "review"), ("flag: wrong size", "wrong size"), ("flag: absent", "absent")):
        text = f"{counts.get(key, 0)} {name}"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x -= tw + 28
        cv2.circle(bar, (x, 22), 6, BGR[key], -1)
        cv2.putText(bar, text, (x + 12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--fps", type=int, default=10)
    args = ap.parse_args()
    s = args.scene
    args.out.mkdir(exist_ok=True)
    cam = json.loads((s / "render" / "cameras.json").read_text())
    comps = {c["id"]: c for c in json.loads((s / "parsed_design.json").read_text())["components"]}
    verdicts = [(comps[r["id"]], outcome(r)) for r in json.loads((s / "verify.json").read_text())["members"]]
    counts = Counter(o for _, o in verdicts)
    frames = sorted((s / "render" / "frames").glob("frame_*.jpg"))
    video, gif = [], []
    for k, fp in enumerate(frames):
        f = int(fp.stem.split("_")[1])
        img = cv2.imread(str(fp))
        for c, o in verdicts:
            if o == "passed":
                draw_member(img, c, BGR[o], cam, f, thick=False)
        for c, o in verdicts:
            if o in LABEL:
                draw_member(img, c, BGR[o], cam, f, thick=True, label=f"{c['id']}  {LABEL[o]}")
        img = header(img, s.name, counts)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        video.append(rgb)
        if k % 3 == 0:
            gif.append(Image.fromarray(rgb).resize((640, round(rgb.shape[0] * 640 / rgb.shape[1])), Image.LANCZOS)
                       .convert("P", palette=Image.ADAPTIVE, colors=160))
    mp4 = args.out / f"replay_{s.name}.mp4"
    with imageio.get_writer(mp4, fps=args.fps, codec="libx264", quality=8, macro_block_size=1) as w:
        for fr in video:
            w.append_data(fr)
    g = args.out / f"replay_{s.name}.gif"
    gif[0].save(g, save_all=True, append_images=gif[1:], duration=int(3000 / args.fps), loop=0, optimize=True)
    print(f"{mp4}: {mp4.stat().st_size / 1e6:.1f} MB\n{g}: {g.stat().st_size / 1e6:.1f} MB\nflagged: "
          + ", ".join(f"{c['id']} ({LABEL[o]})" for c, o in verdicts if o in LABEL))


if __name__ == "__main__":
    main()
