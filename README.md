# FerroSight

Industrial visual intelligence: compare what the engineering drawings specify against what
site footage actually shows, and report each discrepancy with visual evidence.

**Status: phase 1 of 7 (synthetic data generator) is complete.**

## Why synthetic data

Paired drawings + site video + labelled discrepancies do not exist publicly. So FerroSight
generates them: a parametric steel frame is designed, drawn, exported to IFC, then "built"
with known defects and filmed in Blender. Because the defects are injected, every scene has
exact ground truth, which makes the system measurable instead of just demoable.

## What one scene contains

| File | What it is |
| --- | --- |
| `design.json`, `design.ifc` | The design as a component graph and as a BIM model |
| `drawings/*.png`, `drawings.pdf` | Framing plans, frame elevations, notes + member schedule |
| `drawings/annotations.json` | Pixel location of every member and tag on every sheet |
| `asbuilt.json`, `asbuilt.ifc` | What was actually built (design + defects) |
| `defects.json` | Injected discrepancies: `missing`, `wrong_section`, `misplaced`, `extra` |
| `render/frames/*.jpg`, `site_walkthrough.mp4` | Site walkthrough footage |
| `render/gt.npz`, `render/cameras.json` | Per-frame instance masks, depth, camera poses |
| `labels.json` | Benchmark labels, including which components were actually observable |

`labels.json` gives each design component an `expected_verdict`. If a component (or the spot
where a missing one belongs) never appears unoccluded in the footage, the correct answer is
`not_visible`, not a guess. Scenes with zero defects are generated too, to measure false alarms.

## Run it

```bash
pip install -r requirements.txt
pytest -q

# designs, drawings and IFC only (no Blender needed)
python -m ferrosight.datagen.generate --out data/synth --n 20

# full scenes with rendered footage and labels (needs Blender 4.x on PATH or --blender /path/to/blender)
python -m ferrosight.datagen.generate --out data/synth --n 20 --render

# render scenes that were generated earlier
python -m ferrosight.datagen.generate --out data/synth --n 20 --render-only
```

Useful flags: `--frames 120 --width 960 --height 540 --samples 48 --device AUTO|GPU|CPU --defects N`.

## Roadmap

1. Synthetic data generator (done)
2. Drawing parser: sheets -> design graph
3. Site perception: video -> as-built inventory with evidence frames
4. Alignment and diff
5. Verifier: confirmed / refuted / not visible
6. Web app with side-by-side evidence
7. Benchmark results and real-footage validation
