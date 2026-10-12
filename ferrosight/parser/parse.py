"""CLI: parse the drawing set of one or more scenes into parsed_design.json."""
from __future__ import annotations

import argparse
from pathlib import Path

from .assemble import parse_scene
from .ocr import SheetOCR


def main() -> None:
    p = argparse.ArgumentParser(description="FerroSight drawing parser")
    p.add_argument("scenes", nargs="+", type=Path, help="scene folders (each with a drawings/ folder)")
    p.add_argument("--overlays", action="store_true", help="also write annotated sheets to parsed_overlays/")
    args = p.parse_args()
    ocr = SheetOCR(tile=4000, overlap=0, scale=1.0)
    for scene in args.scenes:
        out = parse_scene(scene, ocr, overlays=args.overlays)
        print(f"{scene.name}: {out['stats']['components']} components in {out['stats']['seconds']} s, "
              f"{len(out['warnings'])} warning(s)")
        for w in out["warnings"][:10]:
            print(f"   - {w}")


if __name__ == "__main__":
    main()
