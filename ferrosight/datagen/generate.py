"""Dataset generator CLI.

    python -m ferrosight.datagen.generate --out data/synth --n 20              # designs, drawings, IFC
    python -m ferrosight.datagen.generate --out data/synth --n 20 --render     # + rendered walkthroughs and labels
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .defects import inject_defects
from .drawings import render_drawings
from .geometry import profile_outline
from .postprocess import build_labels, write_video
from .structure import generate_design

RENDER_SCRIPT = Path(__file__).with_name("blender_render.py")


def build_scene(seed: int, out: Path, n_defects: int | None = None) -> Path:
    scene_dir = out / f"scene_{seed:04d}"
    scene_dir.mkdir(parents=True, exist_ok=True)
    design = generate_design(seed)
    built, defects = inject_defects(design, seed, n_defects)
    design.save(scene_dir / "design.json")
    (scene_dir / "defects.json").write_text(json.dumps(defects, indent=1))
    asbuilt = dict(name=design.name, grid_x=design.grid_x, grid_y=design.grid_y, levels=design.levels,
                   components=[asdict(c) | {"outline": profile_outline(c.section)} for c in built])
    (scene_dir / "asbuilt.json").write_text(json.dumps(asbuilt))
    render_drawings(design, scene_dir)
    try:  # IFC export is optional; nothing downstream depends on it
        from .ifc_export import export_ifc
        export_ifc(design, scene_dir / "design.ifc")
        export_ifc(design, scene_dir / "asbuilt.ifc", components=built)
    except ImportError:
        pass
    return scene_dir


def render_scene(scene_dir: Path, args) -> None:
    blender = shutil.which(args.blender) or args.blender
    cmd = [blender, "-b", "-P", str(RENDER_SCRIPT), "--", "--scene", str(scene_dir / "asbuilt.json"),
           "--out", str(scene_dir / "render"), "--frames", str(args.frames), "--width", str(args.width),
           "--height", str(args.height), "--samples", str(args.samples), "--device", args.device,
           "--mask-scale", str(args.mask_scale), "--seed", str(int(scene_dir.name.split("_")[-1]))]
    log = scene_dir / "render.log"
    with open(log, "w") as fh:
        res = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    if res.returncode != 0 or not (scene_dir / "render" / "gt.npz").exists():
        sys.exit(f"Blender render failed for {scene_dir.name}; see {log}")
    labels = build_labels(scene_dir)
    write_video(scene_dir)
    print(f"  labels: {labels['summary']}")


def main() -> None:
    p = argparse.ArgumentParser(description="FerroSight synthetic dataset generator")
    p.add_argument("--out", type=Path, default=Path("data/synth"))
    p.add_argument("--n", type=int, default=5, help="number of scenes")
    p.add_argument("--start-seed", type=int, default=0)
    p.add_argument("--defects", type=int, default=None, help="force a defect count (default: random 0-5)")
    p.add_argument("--render", action="store_true", help="render walkthroughs with Blender and build labels")
    p.add_argument("--render-only", action="store_true", help="skip design generation, render existing scenes")
    p.add_argument("--blender", default="blender", help="path to the Blender executable")
    p.add_argument("--frames", type=int, default=120)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--samples", type=int, default=48)
    p.add_argument("--mask-scale", type=int, default=4)
    p.add_argument("--device", default="AUTO", choices=["AUTO", "GPU", "CPU"])
    args = p.parse_args()

    for seed in range(args.start_seed, args.start_seed + args.n):
        scene_dir = args.out / f"scene_{seed:04d}"
        if not args.render_only:
            scene_dir = build_scene(seed, args.out, args.defects)
            n_def = len(json.loads((scene_dir / "defects.json").read_text()))
            print(f"{scene_dir.name}: design + drawings + IFC written ({n_def} defects)")
        if args.render or args.render_only:
            print(f"{scene_dir.name}: rendering {args.frames} frames ...")
            render_scene(scene_dir, args)


if __name__ == "__main__":
    main()
