"""Multimodal judge: ask a vision-language model the questions FerroSight answers, and score it.

  read   drawing crop only: what section is written for the boxed member?  (can VLMs read drawings?)
  judge  drawing crop + two site frames with the member's expected position outlined:
         is it there, and is it the drawn size?

All markings are drawn in neutral blue, so the images never reveal FerroSight's own verdict.
Asked about: every planted defect, plus N_OK correct members per scene (for the false-alarm rate).

    python -m ferrosight.vlm.judge data/synth --model Qwen/Qwen2.5-VL-3B-Instruct
    python -m ferrosight.vlm.judge data/synth --model Qwen/Qwen2.5-VL-3B-Instruct --score-only
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ..matching.match import project
from ..parser.text import match_section
from ..verify.verify import outcome

N_OK = 20
BLUE = (230, 120, 0)          # BGR
MAX_SIDE = 768

READ_PROMPT = ("This is a crop of a structural steel drawing. One member is marked with a blue box. "
               "What steel section size is written in that member's label (for example W16x40 or HSS6x6x3/8)? "
               "Reply with the section only.")
JUDGE_PROMPT = ("Image 1 is part of a structural steel drawing; the member inside the blue box is a {kind}, "
                "drawn as section {section}. Images 2 and 3 are photos of the building on site; the thin blue "
                "line marks where that member should be if it was built as drawn. Look at the steel along the "
                "blue line. Reply with JSON only: "
                '{{"present": "yes" | "no" | "cannot_tell", '
                '"size_vs_drawing": "smaller" | "same" | "larger" | "cannot_tell", "reason": "<one sentence>"}}')


def shrink(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def drawing_crop(scene: Path, comp) -> Image.Image | None:
    evs = sorted([e for e in comp.get("evidence", []) if "line_px" in e or "tag_bbox_px" in e], key=lambda e: (not e["sheet"].startswith("S-1"), "line_px" not in e))
    if not evs:
        return None
    e = evs[0]
    img = cv2.imread(str(scene / "drawings" / f"{e['sheet']}.png"))
    boxes = []
    if "line_px" in e:
        x0, y0, x1, y1 = e["line_px"]
        boxes.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    if "tag_bbox_px" in e:
        boxes.append(e["tag_bbox_px"])
    b = np.array(boxes, float)
    X0, Y0, X1, Y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    cv2.rectangle(img, (int(X0) - 8, int(Y0) - 8), (int(X1) + 8, int(Y1) + 8), BLUE, 3)
    H, W = img.shape[:2]
    pad = 160
    crop = img[max(0, int(Y0) - pad):min(H, int(Y1) + pad), max(0, int(X0) - pad):min(W, int(X1) + pad)]
    return shrink(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))


def frame_view(scene: Path, cam, comp, f: int) -> Image.Image | None:
    img = cv2.imread(str(scene / "render" / "frames" / f"frame_{f:04d}.jpg"))
    H, W = img.shape[:2]
    a, b = np.asarray(comp["start"], float), np.asarray(comp["end"], float)
    if comp["kind"] == "baseplate":
        mid = (a + b) / 2
        pts = mid + np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0], [-.25, .25, 0], [-.25, -.25, 0]])
    else:
        pts = a + np.linspace(0, 1, 24)[:, None] * (b - a)
    u, v, z = project(pts, cam["frames"][f]["cam_to_world"], cam)
    front = z > 0.2
    if front.sum() < 2:
        return None
    P = np.round(np.c_[u[front], v[front]]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [P], False, BLUE, 2, cv2.LINE_AA)
    uu, vv = u[front], v[front]
    m = 0.7 * max(np.ptp(uu), np.ptp(vv)) + 80
    x0, y0 = int(max(0, uu.min() - m)), int(max(0, vv.min() - m))
    x1, y1 = int(min(W, uu.max() + m)), int(min(H, vv.max() + m))
    if x1 - x0 < 64 or y1 - y0 < 64:
        return None
    return shrink(Image.fromarray(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)))


class VLM:
    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch = torch
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, images, text: str, max_new_tokens: int = 160) -> str:
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
        inputs = self.proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                               tokenize=True, return_dict=True, return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def parse_judge(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group())
            return dict(present=str(d.get("present", "cannot_tell")).lower(),
                        size=str(d.get("size_vs_drawing", "cannot_tell")).lower(), reason=d.get("reason", ""))
        except json.JSONDecodeError:
            pass
    t = text.lower()
    present = "no" if re.search(r'"?present"?\s*:\s*"?no', t) else "yes" if re.search(r'"?present"?\s*:\s*"?yes', t) else "cannot_tell"
    size = next((s for s in ("smaller", "larger", "same") if s in t), "cannot_tell")
    return dict(present=present, size=size, reason=text[:200])


def vlm_outcome(r) -> str:
    if r["present"] == "no":
        return "flag: absent"
    if r["present"] == "yes" and r["size"] in ("smaller", "larger"):
        return "flag: wrong size"
    if r["present"] == "yes":
        return "passed"
    return "not visible"


def pick_members(scene: Path, rng) -> list[str]:
    labels = json.loads((scene / "labels.json").read_text())["components"]
    match = {m["id"]: m for m in json.loads((scene / "matching.json").read_text())["members"]}
    defects = [i for i, l in labels.items() if l["status"] != "ok" and i in match]
    ok = [i for i, l in labels.items() if l["status"] == "ok" and len(match.get(i, {}).get("evidence", [])) >= 2]
    return defects + rng.sample(ok, min(N_OK, len(ok)))


def run(data: Path, model_id: str, out_path: Path):
    vlm = VLM(model_id)
    rng = random.Random(0)
    records = []
    t0 = time.time()
    for scene in sorted(s for s in data.glob("scene_*") if (s / "verify.json").exists()):
        cam = json.loads((scene / "render" / "cameras.json").read_text())
        comps = {c["id"]: c for c in json.loads((scene / "parsed_design.json").read_text())["components"]}
        truth_sec = {c["id"]: c["section"] for c in json.loads((scene / "design.json").read_text())["components"]}
        labels = json.loads((scene / "labels.json").read_text())["components"]
        match = {m["id"]: m for m in json.loads((scene / "matching.json").read_text())["members"]}
        geo = {r["id"]: outcome(r) for r in json.loads((scene / "verify.json").read_text())["members"]}
        for mid in pick_members(scene, rng):
            c = comps.get(mid)
            if c is None:
                continue
            rec = dict(scene=scene.name, id=mid, kind=c["kind"], truth=labels[mid]["status"],
                       observable=labels[mid]["observable"], geo=geo.get(mid, "not visible"),
                       design_section=truth_sec.get(mid))
            draw = drawing_crop(scene, c)
            if draw is not None:
                rec["read_raw"] = vlm.ask([draw], READ_PROMPT, max_new_tokens=24)
                rec["read_section"] = match_section(rec["read_raw"])[0]
            frames = [ev["frame"] for ev in match[mid].get("evidence", [])]
            views = [v for v in (frame_view(scene, cam, c, f) for f in frames) if v is not None][:2]
            if draw is not None and views:
                raw = vlm.ask([draw] + views, JUDGE_PROMPT.format(kind=c["kind"], section=c["section"]))
                rec.update(judge_raw=raw, **parse_judge(raw))
            records.append(rec)
        print(f"{scene.name}: {len(records)} questions so far ({time.time() - t0:.0f} s)", flush=True)
    out_path.write_text(json.dumps(dict(model=model_id, records=records), indent=1))


def score(out_path: Path) -> str:
    d = json.loads(out_path.read_text())
    recs = d["records"]
    read = [r for r in recs if "read_raw" in r]
    read_ok = sum(r.get("read_section") == r["design_section"] for r in read)
    judged = [r for r in recs if "present" in r]

    def system(r, name):
        if name == "Geometric (FerroSight)":
            return r["geo"]
        if name == "VLM only":
            return vlm_outcome(r)
        o = r["geo"]
        if o == "review":
            v = vlm_outcome(r)
            return v if v != "not visible" else "review"
        return o

    want = {"missing": "flag: absent", "misplaced": "flag: absent", "wrong_section": "flag: wrong size"}
    lines = [f"Model: {d['model']}", "",
             f"Reading drawings: {read_ok}/{len(read)} sections read exactly ({read_ok / max(1, len(read)):.1%}). "
             "FerroSight's parser: 100%.", "",
             f"Judging {len(judged)} members (all planted defects + a sample of correct members):", "",
             "| System | Missing caught | Moved caught | Wrong size caught | False alarms on correct members | Left for review |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name in ("Geometric (FerroSight)", "VLM only", "Hybrid (VLM decides the review cases)"):
        caught, seen = Counter(), Counter()
        fa = ok = review = 0
        for r in judged:
            o = system(r, name)
            review += o == "review"
            if r["truth"] == "ok":
                ok += 1
                fa += o.startswith("flag")
            elif r["observable"]:
                seen[r["truth"]] += 1
                caught[r["truth"]] += o == want[r["truth"]]
        cell = lambda t: f"{caught[t]}/{seen[t]}"
        lines.append(f"| {name} | {cell('missing')} | {cell('misplaced')} | {cell('wrong_section')} | "
                     f"{fa}/{ok} ({fa / max(1, ok):.1%}) | {review} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--score-only", action="store_true")
    args = ap.parse_args()
    out = args.data / f"vlm_{args.model.split('/')[-1]}.json"
    if not args.score_only:
        run(args.data, args.model, out)
    report = score(out)
    print("\n" + report)
    (args.data / f"vlm_{args.model.split('/')[-1]}.md").write_text(report + "\n")


if __name__ == "__main__":
    main()
