"""Score SAM 3 steel masks against the renderer's exact answer masks.

    python -m ferrosight.perception.evaluate data/synth [--threshold 0.4]

Reports pixel overlap with the true steel, the share of visible members of each type that SAM 3
covered (a member counts once per frame it is visible in), and how often each prompt's
detections were actually steel. Writes data/synth/perception_eval.md.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MIN_PIXELS = 4      # cells at answer-mask resolution (4 x 4 image pixels each)
COVERED = 0.5       # share of a member's pixels SAM 3 must cover for it to count as found


def score_scene(scene: Path, thr: float) -> dict:
    gt = np.load(scene / "render" / "gt.npz")["masks"]
    ids = json.loads((scene / "render" / "cameras.json").read_text())["id_map"]
    kind_of = {c["id"]: c["kind"] for c in json.loads((scene / "asbuilt.json").read_text())["components"]}
    meta = json.loads((scene / "perception" / "sam3.json").read_text())
    p = np.load(scene / "perception" / "sam3.npz")
    masks, frames, pidx, scores = p["masks"], p["frame"], p["prompt"], p["score"]
    keep = scores >= thr
    r = dict(tp=0, fp=0, fn=0, kinds=defaultdict(lambda: [0, 0]), prompts=defaultdict(lambda: [0, 0]))
    for f in meta["frames"]:
        g = gt[f]
        steel = g > 0
        sel = keep & (frames == f)
        pred = masks[sel].any(0) if sel.any() else np.zeros_like(steel)
        r["tp"] += int((pred & steel).sum())
        r["fp"] += int((pred & ~steel).sum())
        r["fn"] += int((~pred & steel).sum())
        for i in np.unique(g):
            if i == 0:
                continue
            m = g == i
            if m.sum() < MIN_PIXELS:
                continue
            k = kind_of.get(ids[i], "other")
            r["kinds"][k][0] += 1
            r["kinds"][k][1] += int(pred[m].mean() >= COVERED)
        for j in np.nonzero(sel)[0]:
            a = masks[j].sum()
            if a == 0:
                continue
            name = meta["prompts"][pidx[j]]
            r["prompts"][name][0] += 1
            r["prompts"][name][1] += int((masks[j] & steel).sum() / a >= 0.5)
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--threshold", type=float, default=0.4)
    args = ap.parse_args()
    scenes = sorted(s for s in args.data.glob("scene_*") if (s / "perception" / "sam3.npz").exists())
    if not scenes:
        print("no scenes with perception/sam3.npz; run ferrosight.perception.segment first")
        return
    tot = dict(tp=0, fp=0, fn=0, kinds=defaultdict(lambda: [0, 0]), prompts=defaultdict(lambda: [0, 0]))
    for s in scenes:
        r = score_scene(s, args.threshold)
        for k in ("tp", "fp", "fn"):
            tot[k] += r[k]
        for k, (a, b) in r["kinds"].items():
            tot["kinds"][k][0] += a
            tot["kinds"][k][1] += b
        for k, (a, b) in r["prompts"].items():
            tot["prompts"][k][0] += a
            tot["prompts"][k][1] += b
        iou = r["tp"] / max(1, r["tp"] + r["fp"] + r["fn"])
        print(f"{s.name}: steel IoU {iou:.1%}")
    tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
    lines = [f"SAM 3 on {len(scenes)} scenes, score threshold {args.threshold}", "",
             "| Steel pixels | IoU | Precision | Recall |", "| --- | ---: | ---: | ---: |",
             f"| all | {tp / max(1, tp + fp + fn):.1%} | {tp / max(1, tp + fp):.1%} | {tp / max(1, tp + fn):.1%} |",
             "", "| Member type | Visible (member-frames) | Covered by SAM 3 |", "| --- | ---: | ---: |"]
    for k, (a, b) in sorted(tot["kinds"].items()):
        lines.append(f"| {k} | {a} | {b / max(1, a):.1%} |")
    lines += ["", "| Prompt | Detections | Actually steel |", "| --- | ---: | ---: |"]
    for k, (a, b) in tot["prompts"].items():
        lines.append(f"| {k} | {a} | {b / max(1, a):.1%} |")
    report = "\n".join(lines)
    print("\n" + report)
    (args.data / "perception_eval.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
