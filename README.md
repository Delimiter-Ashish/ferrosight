# FerroSight

Industrial visual intelligence: compare what the engineering drawings specify against what
site footage actually shows, and report each discrepancy with visual evidence.

**Status: phases 1 and 2 of 7 are complete (synthetic data generator, drawing parser).**

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

## Drawing parser (phase 2)

Reads only the sheet images (like scanned drawings) and rebuilds the design: grid, levels, and
every member with its section, mark, 3D position, a confidence score, and where on which sheet it
was found. Output goes to `parsed_design.json` in the scene folder.

```bash
# parse one scene (add --overlays to also save annotated sheets)
python -m ferrosight.parser.parse data/synth/scene_0000 --overlays

# parse every scene and score it against the generator's ground truth
python -m ferrosight.parser.evaluate data/synth
python -m ferrosight.parser.evaluate data/synth --damage light   # sheets degraded like a scan
python -m ferrosight.parser.evaluate data/synth --damage heavy
```

The score table is written to `data/synth/parser_eval*.md`.

### Results (20 synthetic buildings, 1,790 members)

A member counts as found only if the parser puts it in the right place (both ends within 30 cm),
so a correct label alone is not enough.

| Drawings | Found | Spurious | Section correct | Mark correct | Mean position error | Time per building |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clean | 100% | 0 | 100% | 100% | 1.0 cm | 9 s |
| Light scan damage (blur, noise, fading, up to 0.35° skew) | 100% | 0 | 100% | 100% | 1.1 cm | 9 s |
| Heavy scan damage | parse fails | | | | | |

How it works: sheets are cleaned up (contrast, deskew), linework is extracted (column squares,
solid and dashed lines, diagonals) and erased so it cannot corrupt the text, the remaining text is
read by OCR, and each sheet is interpreted in its own coordinates (gridlines and dimension
strings on plans, level annotations on elevations). Readings are then merged across sheets:
sections are voted on, every mark is checked against the mark its position implies, and member
counts are reconciled with the schedule on S-001. Disagreements become warnings, not silent fixes.

Known limits: on heavily damaged sheets the OCR cannot read titles and dimension strings, so the
parser stops rather than guess. Grid bubble letters are not read by the OCR model, so grid labels
follow the standard order (numbers left to right, letters bottom to top). The parser is built for
this drawing set's conventions.

## Roadmap

1. Synthetic data generator (done)
2. Drawing parser: sheets -> design graph (done)
3. Site perception: video -> as-built inventory with evidence frames
4. Alignment and diff
5. Verifier: confirmed / refuted / not visible
6. Web app with side-by-side evidence
7. Benchmark results and real-footage validation
