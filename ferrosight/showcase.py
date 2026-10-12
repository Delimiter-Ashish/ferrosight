"""Build the README images from processed scenes.

    python -m ferrosight.showcase --data data/synth --out docs

Writes docs/hero.gif, docs/pipeline.png and docs/results.png. Images come from the same
code paths as the web app, so the README shows exactly what the tool produces.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi.testclient import TestClient
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .app.server import create_app
from .verify.verify import outcome

DARK, ORANGE, WHITE = (28, 29, 31), (255, 120, 40), (255, 255, 255)


def font(size, bold=True):
    path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal"))
    return ImageFont.truetype(path, size)


def get_img(client, url):
    r = client.get(url)
    return Image.open(io.BytesIO(r.content)).convert("RGB") if r.status_code == 200 else None


def fit_h(img, h):
    return img.resize((max(1, round(img.width * h / img.height)), h), Image.LANCZOS)


def pick_example(client):
    best = None
    for s in client.get("/api/scenes").json():
        for m in client.get(f"/api/scenes/{s['name']}").json()["members"]:
            if m["outcome"] == "flag: absent" and m["truth"] in ("missing", "misplaced") and m["has_drawing"] and m["frames"]:
                score = (m["kind"] == "brace", m["truth"] == "missing", m["clear_views"])
                if best is None or score > best[0]:
                    best = (score, s["name"], m)
    if best is None:
        raise SystemExit("no correctly flagged absent member with a drawing found")
    return best[1], best[2]


def hero(client, data, scene, m, out):
    enc = m["id"]
    drawing = get_img(client, f"/api/scenes/{scene}/drawing/{enc}.png")
    frames = json.loads((data / scene / "perception" / "sam3.json").read_text())["frames"]
    shots = []
    for f in list(dict.fromkeys(m["frames"] + frames)):
        img = get_img(client, f"/api/scenes/{scene}/frame/{f}/{enc}.jpg")
        if img is not None:
            shots.append(img)
        if len(shots) == 10:
            break
    title = f"{m['id']}: the drawings show a {m['kind']} here."
    sub = "Site footage: no steel in any clear view.  ->  FLAGGED: ABSENT"
    frames_out = []
    LW, RW, H = 440, 860, 400

    def boxed(img, bw, bh):
        img = img.copy()
        img.thumbnail((bw, bh), Image.LANCZOS)
        box = Image.new("RGB", (bw, bh), DARK)
        box.paste(img, ((bw - img.width) // 2, (bh - img.height) // 2))
        return box
    left = boxed(drawing, LW, H)
    for shot in shots:
        canvas = Image.new("RGB", (LW + RW + 30, H + 100), DARK)
        canvas.paste(left, (10, 90))
        canvas.paste(boxed(shot, RW, H), (LW + 20, 90))
        d = ImageDraw.Draw(canvas)
        d.text((14, 14), title, font=font(26), fill=WHITE)
        d.text((14, 52), sub, font=font(18), fill=ORANGE)
        frames_out.append(canvas.convert("P", palette=Image.ADAPTIVE, colors=200))
    frames_out[0].save(out / "hero.gif", save_all=True, append_images=frames_out[1:], duration=900, loop=0, optimize=True)


def pipeline(client, data, scene, m, out):
    design = json.loads((data / scene / "parsed_design.json").read_text())
    comp = next(c for c in design["components"] if c["id"] == m["id"])
    ev = sorted(comp["evidence"], key=lambda e: (not e["sheet"].startswith("S-1"), "line_px" not in e))[0]
    f = m["frames"][0]
    sheet = Image.open(data / scene / "drawings" / f"{ev['sheet']}.png").convert("RGB")
    raw = Image.open(data / scene / "render" / "frames" / f"frame_{f:04d}.jpg").convert("RGB")
    p = np.load(data / scene / "perception" / "sam3.npz")
    meta = json.loads((data / scene / "perception" / "sam3.json").read_text())
    sel = (p["frame"] == f) & (p["score"] >= meta["threshold"])
    mask = p["masks"][sel].any(0) if sel.any() else np.zeros(meta["mask_shape"], bool)
    mask = cv2.resize(mask.astype(np.uint8), raw.size, interpolation=cv2.INTER_NEAREST).astype(bool)
    tinted = np.asarray(raw).copy()
    tinted[mask] = (0.4 * tinted[mask] + 0.6 * np.array(ORANGE)).astype(np.uint8)
    panels = [("1  Drawing set", sheet), ("2  Member read from drawings", get_img(client, f"/api/scenes/{scene}/drawing/{m['id']}.png")),
              ("3  Site footage", raw), ("4  Steel found by SAM 3", Image.fromarray(tinted)),
              ("5  Verdict: absent", get_img(client, f"/api/scenes/{scene}/frame/{f}/{m['id']}.jpg"))]
    H = 260
    imgs = [fit_h(img, H) for _, img in panels]
    canvas = Image.new("RGB", (sum(i.width for i in imgs) + 12 * (len(imgs) + 1), H + 60), WHITE)
    x = 12
    d = ImageDraw.Draw(canvas)
    for (label, _), img in zip(panels, imgs):
        canvas.paste(img, (x, 12))
        d.text((x, H + 22), label, font=font(17), fill=DARK)
        x += img.width + 12
    canvas.save(out / "pipeline.png")


def results(data, out):
    caught, seen, ok_flagged, ok_total = Counter(), Counter(), 0, 0
    for s in sorted(data.glob("scene_*")):
        if not (s / "verify.json").exists():
            continue
        labels = json.loads((s / "labels.json").read_text())["components"]
        for r in json.loads((s / "verify.json").read_text())["members"]:
            lab = labels.get(r["id"])
            if not lab:
                continue
            o = outcome(r)
            if lab["status"] == "ok":
                ok_total += 1
                ok_flagged += o.startswith("flag")
            elif lab["observable"]:
                kind = {"misplaced": "Moved member", "missing": "Missing member", "wrong_section": "Wrong size"}[lab["status"]]
                seen[kind] += 1
                caught[kind] += (o == "flag: absent") if kind != "Wrong size" else (o == "flag: wrong size")
    kinds = ["Missing member", "Moved member", "Wrong size"]
    vals = [caught[k] / max(1, seen[k]) * 100 for k in kinds]
    fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=150)
    bars = ax.barh(kinds, vals, color="#ff7828", height=0.55)
    for b, k in zip(bars, kinds):
        ax.text(b.get_width() + 1.5, b.get_y() + b.get_height() / 2, f"{caught[k]}/{seen[k]}", va="center", fontsize=11)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Mistakes caught (%), counting only mistakes the camera saw")
    ax.invert_yaxis()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"20 test buildings. False alarms: {ok_flagged} of {ok_total} correct members "
                 f"({ok_flagged / max(1, ok_total):.2%})", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(out / "results.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synth"))
    ap.add_argument("--out", type=Path, default=Path("docs"))
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True)
    client = TestClient(create_app(args.data))
    scene, m = pick_example(client)
    print(f"example: {scene} {m['id']} ({m['truth']})")
    hero(client, args.data, scene, m, args.out)
    pipeline(client, args.data, scene, m, args.out)
    results(args.data, args.out)
    for f in ("hero.gif", "pipeline.png", "results.png"):
        print(f"{args.out / f}: {(args.out / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
