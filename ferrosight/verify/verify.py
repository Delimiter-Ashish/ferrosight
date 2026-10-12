"""Phase 5: verify members by looking closer.

The matcher works on low-resolution masks. Here, for each member that needs a decision, the best
views are cropped around the member, enlarged, and segmented again with SAM 3 at full detail.

  presence  members the matcher called absent or inconclusive: share of the member's clean sample
            points (no other design member on the same pixels) that land on steel in the close-up
  size      present girders, beams, joists and braces: steel thickness measured across the member,
            compared with the thickness each catalogue section would show from that viewpoint

    python -m ferrosight.verify.verify data/synth --evaluate          # run + score
    python -m ferrosight.verify.verify data/synth --evaluate --no-run # score existing verify.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..datagen.geometry import local_frame, profile_outline
from ..datagen.sections import SECTIONS
from ..matching.match import project, sample_points, width_of

CROP_SIDE = 1008          # close-ups are enlarged to about this many pixels on the long side
MAX_ZOOM = 4.0
SIZE_KINDS = ("girder", "beam", "joist", "brace")
SIZE_TOL = math.log(1.2)
THIN_FLAG = 0.74        # measured/drawn below this: flag (learned on scenes 0-9)
THICK_REVIEW = 1.45     # above this: review; outlines spill onto neighbouring steel, so "thicker" is never flagged outright  # measured / expected thickness within 20% counts as a fit
MIN_MEASURES = 4
K5 = np.ones((5, 5), np.uint8)


def crop_box(bbox, W, H):
    x0, y0, x1, y1 = bbox
    m = 0.6 * max(x1 - x0, y1 - y0) + 24
    return max(0, int(x0 - m)), max(0, int(y0 - m)), min(W, int(x1 + m)), min(H, int(y1 + m))


def closeup_K(K, x0, y0, s):
    return dict(fx=K["fx"] * s, fy=K["fy"] * s, cx=(K["cx"] - x0) * s, cy=(K["cy"] - y0) * s)


def buffers(comps, c2w, K, h, w):
    """Nearest design member per pixel, and how many members (grown by 2 px) reach each pixel."""
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
        u, v = u[front], v[front]
        if u.max() < -50 or u.min() > w + 50 or v.max() < -50 or v.min() > h + 50:
            continue                                           # nowhere near this close-up
        order.append((float(z[front].mean()), i, np.clip(u, -2e4, 2e4), np.clip(v, -2e4, 2e4)))
    for zm, i, u, v in sorted(order, key=lambda t: -t[0]):
        fp = np.zeros((h, w), np.uint8)
        pts = np.round(np.c_[u, v]).astype(np.int32).reshape(-1, 1, 2)
        if comps[i]["kind"] == "baseplate":
            cv2.fillConvexPoly(fp, pts, 1)
        else:
            t = int(np.clip(round(K["fx"] * width_of(comps[i]) / max(zm, 0.1)), 1, 800))
            cv2.polylines(fp, [pts], False, 1, thickness=t)
        index[fp > 0] = i
        cover += cv2.dilate(fp, K5)
    return index, cover


def clean_points(c, i, c2w, K, h, w, index, cover):
    pts = sample_points(c)
    u, v, z = project(pts, c2w, K)
    iu, iv = np.round(u).astype(int), np.round(v).astype(int)
    inside = (z > 0.1) & (iu >= 0) & (iu < w) & (iv >= 0) & (iv < h)
    vis = np.zeros(len(u), bool)
    vis[inside] = index[iv[inside], iu[inside]] == i
    clean = vis.copy()
    clean[vis] = cover[iv[vis], iu[vis]] == 1
    return pts, u, v, clean


def run_length(mask, u, v, n, max_len):
    """Steel pixels across the member through (u, v) along direction n. None if not on steel or
    the run reaches the edge of the close-up (no clean edge to measure)."""
    h, w = mask.shape
    if not (0 <= int(round(u)) < w and 0 <= int(round(v)) < h) or not mask[int(round(v)), int(round(u))]:
        return None
    run = 1
    for sgn in (1, -1):
        k = 1
        while True:
            x, y = int(round(u + sgn * k * n[0])), int(round(v + sgn * k * n[1]))
            if not (0 <= x < w and 0 <= y < h) or k > max_len:
                return None
            if not mask[y, x]:
                break
            run += 1
            k += 1
    return run


def expected_thickness(c, section, p3, c2w, K, u0, v0, n):
    x, y, _, _ = local_frame(c["start"], c["end"], c["up"])
    corners = np.array([p3 + x * ox + y * oy for ox, oy in profile_outline(section)])
    uu, vv, zz = project(corners, c2w, K)
    if (zz <= 0.1).any():
        return None
    d = (uu - u0) * n[0] + (vv - v0) * n[1]
    return float(d.max() - d.min())


def candidates(section):
    shape = SECTIONS[section]["shape"]
    return [s for s, d in SECTIONS.items() if d["shape"] == shape and not s.startswith("PL")]


def verify_scene(scene: Path, proc, model, device, max_views: int) -> dict:
    from ..perception.segment import segment
    design = json.loads((scene / "parsed_design.json").read_text())
    matching = json.loads((scene / "matching.json").read_text())
    cam = json.loads((scene / "render" / "cameras.json").read_text())
    comps = design["components"]
    idx_of = {c["id"]: i for i, c in enumerate(comps)}
    W, H = cam["width"], cam["height"]
    out = []
    t0 = time.time()
    for m in matching["members"]:
        i = idx_of[m["id"]]
        c = comps[i]
        need_presence = m["verdict"] in ("absent", "inconclusive")
        need_size = m["verdict"] == "present" and c["kind"] in SIZE_KINDS and c.get("section") in SECTIONS
        rec = dict(id=c["id"], kind=c["kind"], section=c.get("section"), grid=c.get("grid"),
                   match_verdict=m["verdict"], verdict=m["verdict"], size=None, closeups=[])
        if not (need_presence or need_size):
            out.append(rec)
            continue
        hits, logs = [], defaultdict(list)
        for ev in m["evidence"][:max_views]:
            f = ev["frame"]
            c2w = cam["frames"][f]["cam_to_world"]
            x0, y0, x1, y1 = crop_box(ev["bbox_px"], W, H)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            s = max(1.0, min(MAX_ZOOM, CROP_SIDE / max(x1 - x0, y1 - y0)))
            img = Image.open(scene / "render" / "frames" / f"frame_{f:04d}.jpg").convert("RGB").crop((x0, y0, x1, y1))
            img = img.resize((max(1, round((x1 - x0) * s)), max(1, round((y1 - y0) * s))), Image.BICUBIC)
            w, h = img.size
            K = closeup_K(cam, x0, y0, s)
            prompt = "steel column" if c["kind"] == "column" else "steel beam"
            masks, scores, _ = segment(proc, model, img, prompt, device, 0.4)
            steel = masks.bool().any(0).cpu().numpy() if len(scores) else np.zeros((h, w), bool)
            index, cover = buffers(comps, c2w, K, h, w)
            pts, u, v, clean = clean_points(c, i, c2w, K, h, w, index, cover)
            if clean.sum() < 3:
                continue
            steel_d = cv2.dilate(steel.astype(np.uint8), K5) > 0
            iu, iv = np.round(u).astype(int), np.round(v).astype(int)
            hit = float(steel_d[iv[clean], iu[clean]].mean())
            hits.append(hit)
            rec["closeups"].append(dict(frame=f, crop=[x0, y0, x1, y1], zoom=round(s, 2), steel=round(hit, 2)))
            if need_size:
                a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
                step = 0.05 * (b - a) / np.linalg.norm(b - a)
                cands = candidates(c["section"])
                for j in np.nonzero(clean)[0]:
                    ua, va, _ = project((pts[j] + step)[None], c2w, K)
                    d = np.array([ua[0] - u[j], va[0] - v[j]])
                    if np.linalg.norm(d) < 1e-6:
                        continue
                    d /= np.linalg.norm(d)
                    n = np.array([-d[1], d[0]])
                    exp_drawn = expected_thickness(c, c["section"], pts[j], c2w, K, u[j], v[j], n)
                    if not exp_drawn or exp_drawn < 4:
                        continue
                    t = run_length(steel, u[j], v[j], n, max_len=int(4 * exp_drawn) + 10)
                    if t is None or t < 3:
                        continue
                    for sname in cands:
                        e = expected_thickness(c, sname, pts[j], c2w, K, u[j], v[j], n)
                        if e and e > 0:
                            logs[sname].append(math.log(t / e))
        if need_presence and len(hits) >= 2:
            ratio = float(np.mean([x >= 0.4 for x in hits]))
            rec["verdict"] = "present" if ratio >= 0.5 else "absent" if ratio <= 0.2 else "inconclusive"
        if need_size:
            med = {sname: float(np.median(v)) for sname, v in logs.items() if len(v) >= MIN_MEASURES}
            drawn = c["section"]
            if drawn in med:
                best = min(med, key=lambda sname: abs(med[sname]))
                if abs(med[drawn]) <= SIZE_TOL:
                    status = "ok"
                elif best != drawn and abs(med[best]) <= SIZE_TOL:
                    status = "wrong_section"
                else:
                    status = "uncertain"
                rec["size"] = dict(status=status, looks_like=best, ratio_to_drawn=round(math.exp(med[drawn]), 2),
                                   measurements=len(logs[drawn]))
        out.append(rec)
    res = dict(scene=scene.name, members=out, seconds=round(time.time() - t0, 1),
               summary=dict(Counter(r["verdict"] for r in out)),
               size_summary=dict(Counter(r["size"]["status"] for r in out if r["size"])))
    (scene / "verify.json").write_text(json.dumps(res, indent=1))
    return res


def outcome(r) -> str:
    if r["verdict"] == "absent":
        return "flag: absent"
    if r["verdict"] == "not_visible":
        return "not visible"
    if r["verdict"] == "inconclusive":
        return "review"
    if r["size"]:
        ratio = r["size"]["ratio_to_drawn"]
        if ratio <= THIN_FLAG:
            return "flag: wrong size"
        if ratio >= THICK_REVIEW:
            return "review"
    return "passed"


COLS = ["flag: absent", "flag: wrong size", "review", "passed", "not visible"]


def evaluate(scenes) -> str:
    table = defaultdict(Counter)
    size_named = [0, 0]
    for s in scenes:
        labels = json.loads((s / "labels.json").read_text())["components"]
        actual = {d["component_id"]: d["as_built"]["section"] for d in json.loads((s / "defects.json").read_text())
                  if d["type"] == "wrong_section"}
        for r in json.loads((s / "verify.json").read_text())["members"]:
            lab = labels.get(r["id"])
            if lab is None:
                continue
            truth = lab["status"] if lab["observable"] else lab["status"] + " (camera never saw it)"
            o = outcome(r)
            table[truth][o] += 1
            if lab["status"] == "wrong_section" and o == "flag: wrong size":
                size_named[0] += 1
                size_named[1] += int(r["size"]["looks_like"] == actual.get(r["id"]))
    lines = [f"Verification on {len(scenes)} scenes", "", "| Truth | " + " | ".join(COLS) + " |",
             "| --- |" + " ---: |" * len(COLS)]
    for truth in sorted(table):
        lines.append(f"| {truth} | " + " | ".join(str(table[truth][c]) for c in COLS) + " |")
    caught = table["missing"]["flag: absent"] + table["misplaced"]["flag: absent"] + table["wrong_section"]["flag: wrong size"]
    seen = sum(sum(table[t].values()) for t in ("missing", "misplaced", "wrong_section"))
    ok = table["ok"]
    false = ok["flag: absent"] + ok["flag: wrong size"]
    lines += ["", f"Defects the camera saw: {seen}. Flagged correctly: {caught} ({caught / max(1, seen):.1%}). "
              f"Sent to review: {sum(table[t]['review'] for t in ('missing', 'misplaced', 'wrong_section'))}.",
              f"Correct members wrongly flagged: {false} of {sum(ok.values())} ({false / max(1, sum(ok.values())):.2%}); "
              f"sent to review: {ok['review']}.",
              f"Wrong sizes flagged: {size_named[0]}, actual section named correctly: {size_named[1]}."]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Close-up verification of matched members")
    ap.add_argument("data", type=Path)
    ap.add_argument("--views", type=int, default=2, help="close-ups per member")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--no-run", action="store_true", help="only score existing verify.json files")
    args = ap.parse_args()
    scenes = sorted(s for s in args.data.glob("scene_*") if (s / "matching.json").exists())
    if not args.no_run:
        import torch
        from ..perception.segment import load
        torch.set_num_threads(int(os.environ.get("FS_THREADS", "2")))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        proc, model = load(device)
        for s in scenes:
            r = verify_scene(s, proc, model, device, args.views)
            print(f"{s.name}: {r['summary']} size {r['size_summary']} ({r['seconds']} s)", flush=True)
    if args.evaluate:
        report = evaluate([s for s in scenes if (s / "verify.json").exists()])
        print("\n" + report)
        (args.data / "verify_eval.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
