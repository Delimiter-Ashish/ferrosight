"""Phase 3a: find steel in site footage with SAM 3 (text-prompted segmentation).

    python -m ferrosight.perception.segment data/synth/scene_0000 --check   # one frame + overlay image
    python -m ferrosight.perception.segment data/synth/scene_*  --every 4   # every 4th frame of each scene

Writes <scene>/perception/sam3.npz (masks at the answer-mask resolution) and sam3.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MODEL_ID = "facebook/sam3"
DEFAULT_PROMPTS = ["steel beam", "steel column", "diagonal steel brace"]
COLORS = [(255, 60, 60), (60, 140, 255), (255, 200, 0), (0, 220, 120), (220, 0, 220)]


def load(device: str):
    from transformers import Sam3Model, Sam3Processor
    proc = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to(device).eval()
    return proc, model


@torch.inference_mode()
def segment(proc, model, image: Image.Image, prompt: str, device: str, threshold: float):
    inputs = proc(images=image, text=prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    res = proc.post_process_instance_segmentation(
        outputs, threshold=threshold, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist())[0]
    return res["masks"], res["scores"], res["boxes"]


def run_scene(scene: Path, proc, model, device, prompts, every, threshold, check=False):
    cam = json.loads((scene / "render" / "cameras.json").read_text())
    sc = cam["mask_scale"]
    h, w = cam["height"] // sc, cam["width"] // sc
    frames = sorted((scene / "render" / "frames").glob("frame_*.jpg"))
    frames = frames[len(frames) // 2: len(frames) // 2 + 1] if check else frames[::every]
    out = dict(masks=[], frame=[], prompt=[], score=[], box=[])
    t0 = time.time()
    for fp in frames:
        f = int(fp.stem.split("_")[1])
        img = Image.open(fp).convert("RGB")
        overlay = np.asarray(img).astype(np.float32)
        for pi, prompt in enumerate(prompts):
            masks, scores, boxes = segment(proc, model, img, prompt, device, threshold)
            if len(scores) == 0:
                continue
            m = masks.bool()
            if check:
                for mk in m.cpu().numpy():
                    overlay[mk] = 0.45 * overlay[mk] + 0.55 * np.array(COLORS[pi % len(COLORS)])
                print(f"  {prompt!r}: {len(scores)} instance(s), scores {[round(float(s), 2) for s in scores]}")
            # keep thin members: a low-res cell counts if a quarter of it is covered
            small = F.interpolate(m[:, None].float(), size=(h, w), mode="area")[:, 0] > 0.25
            out["masks"].append(small.cpu().numpy())
            out["frame"] += [f] * len(scores)
            out["prompt"] += [pi] * len(scores)
            out["score"] += [float(s) for s in scores]
            out["box"] += boxes.cpu().numpy().tolist()
        if check:
            dest = scene / "perception"
            dest.mkdir(exist_ok=True)
            Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(dest / f"check_frame_{f:04d}.png")
            print(f"  overlay saved to {dest / f'check_frame_{f:04d}.png'}")
    secs = time.time() - t0
    dest = scene / "perception"
    dest.mkdir(exist_ok=True)
    masks = np.concatenate(out["masks"]) if out["masks"] else np.zeros((0, h, w), bool)
    np.savez_compressed(dest / "sam3.npz", masks=masks, frame=np.array(out["frame"], int),
                        prompt=np.array(out["prompt"], int), score=np.array(out["score"], float),
                        box=np.array(out["box"], float).reshape(-1, 4))
    (dest / "sam3.json").write_text(json.dumps(dict(
        model=MODEL_ID, prompts=prompts, threshold=threshold, mask_shape=[h, w],
        frames=[int(p.stem.split("_")[1]) for p in frames], seconds=round(secs, 1)), indent=1))
    print(f"{scene.name}: {len(frames)} frame(s), {len(out['score'])} instance(s), "
          f"{secs / max(1, len(frames) * len(prompts)):.2f} s per frame per prompt")


def main() -> None:
    p = argparse.ArgumentParser(description="SAM 3 steel segmentation on rendered walkthroughs")
    p.add_argument("scenes", nargs="+", type=Path)
    p.add_argument("--prompts", nargs="+", default=DEFAULT_PROMPTS)
    p.add_argument("--every", type=int, default=4, help="use every Nth frame")
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--check", action="store_true", help="one frame per scene, print detections, save overlay")
    args = p.parse_args()
    torch.set_num_threads(int(os.environ.get("FS_THREADS", "2")))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))
    proc, model = load(device)
    for scene in args.scenes:
        if not (scene / "render" / "gt.npz").exists():
            print(f"{scene.name}: not rendered yet, skipped")
            continue
        run_scene(scene, proc, model, device, args.prompts, args.every, args.threshold, args.check)


if __name__ == "__main__":
    main()
