"""Score parsed_design.json against the generator's design.json.

Matching is geometric (same kind, both endpoints within MATCH_TOL), so a component counts as
found because it is in the right place, not because its label was read.

    python -m ferrosight.parser.evaluate data/synth            # parse every scene, then score
    python -m ferrosight.parser.evaluate data/synth --no-parse # score existing parsed_design.json files
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MATCH_TOL = 0.30   # metres
KINDS = ["column", "girder", "beam", "joist", "brace", "baseplate"]


def _dist(a, b) -> float:
    return float(max(np.linalg.norm(np.subtract(a["start"], b["start"])), np.linalg.norm(np.subtract(a["end"], b["end"]))))


def score_scene(scene_dir: Path, parsed_name: str = "parsed_design.json") -> dict:
    gt = json.loads((scene_dir / "design.json").read_text())
    pr = json.loads((scene_dir / parsed_name).read_text())
    ann = json.loads((scene_dir / "drawings" / "annotations.json").read_text())["annotations"]
    printed = {a["component_id"] for a in ann if "[" in a.get("tag_text", "")}   # mark appears on some sheet
    per = {k: defaultdict(float) for k in KINDS}
    errors, misses, extras = [], [], []
    for kind in KINDS:
        g = [c for c in gt["components"] if c["kind"] == kind]
        p = [c for c in pr["components"] if c["kind"] == kind]
        pairs = sorted(((_dist(a, b), i, j) for i, a in enumerate(g) for j, b in enumerate(p)), key=lambda t: t[0])
        used_g, used_p = set(), set()
        for d, i, j in pairs:
            if d > MATCH_TOL or i in used_g or j in used_p:
                continue
            used_g.add(i), used_p.add(j)
            a, b = g[i], p[j]
            s = per[kind]
            s["matched"] += 1
            s["section_ok"] += a["section"] == b["section"]
            s["id_ok"] += a["id"] == b["id"]
            if a["id"] in printed:
                s["printed"] += 1
                s["mark_read_ok"] += b.get("mark_read") == a["id"]
            s["err_sum"] += d
            if a["section"] != b["section"]:
                errors.append(f"{a['id']}: section {b['section']} ({b['section_source']}) vs {a['section']}")
            if a["id"] != b["id"]:
                errors.append(f"{a['id']}: id parsed as {b['id']}")
        per[kind]["gt"] = len(g)
        per[kind]["parsed"] = len(p)
        misses += [g[i]["id"] for i in range(len(g)) if i not in used_g]
        extras += [p[j]["id"] for j in range(len(p)) if j not in used_p]
    grid_err = max(np.abs(np.subtract(gt["grid_x"], pr["grid_x"])).max() if len(gt["grid_x"]) == len(pr["grid_x"]) else 99,
                   np.abs(np.subtract(gt["grid_y"], pr["grid_y"])).max() if len(gt["grid_y"]) == len(pr["grid_y"]) else 99)
    lvl_err = np.abs(np.subtract(gt["levels"], pr["levels"])).max() if len(gt["levels"]) == len(pr["levels"]) else 99
    return dict(scene=scene_dir.name, per_kind={k: dict(v) for k, v in per.items()}, grid_err=float(grid_err),
                level_err=float(lvl_err), misses=misses, extras=extras, errors=errors,
                warnings=len(pr.get("warnings", [])), seconds=pr.get("stats", {}).get("seconds"))


def summarise(results: list[dict]) -> str:
    if not results:
        return "No scene could be parsed."
    tot = {k: defaultdict(float) for k in KINDS}
    for r in results:
        for k in KINDS:
            for f, v in r["per_kind"][k].items():
                tot[k][f] += v
    lines = ["| Member | In design | Found (recall) | Spurious | Section correct | Mark correct | Printed marks read exactly | Mean position error |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    all_ = defaultdict(float)
    for k in KINDS:
        t = tot[k]
        for f, v in t.items():
            all_[f] += v
    for k, t in list(((k, tot[k]) for k in KINDS)) + [("**all**", all_)]:
        if not t["gt"]:
            continue
        m = max(1, t["matched"])
        lines.append(f"| {k} | {int(t['gt'])} | {t['matched'] / t['gt']:.1%} | {int(t['parsed'] - t['matched'])} | "
                     f"{t['section_ok'] / m:.1%} | {t['id_ok'] / m:.1%} | {t['mark_read_ok'] / max(1, t['printed']):.1%} | "
                     f"{100 * t['err_sum'] / m:.1f} cm |")
    ge = max(r["grid_err"] for r in results)
    le = max(r["level_err"] for r in results)
    secs = [r["seconds"] for r in results if r["seconds"]]
    lines.append("")
    lines.append(f"{len(results)} scenes. Worst grid error {100 * ge:.1f} cm, worst level error {100 * le:.1f} cm, "
                 f"median parse time {np.median(secs):.0f} s per scene.")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("data", type=Path)
    p.add_argument("--no-parse", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--damage", choices=["light", "heavy"], default=None,
                   help="degrade every sheet like a scan/photocopy before parsing")
    args = p.parse_args()
    tag = f"_{args.damage}" if args.damage else ""
    parsed_name = f"parsed_design{tag}.json"
    scenes = sorted(d for d in args.data.glob("scene_*") if (d / "design.json").exists())[: args.limit]
    if not args.no_parse:
        from .assemble import parse_scene
        from .ocr import SheetOCR
        ocr = SheetOCR(tile=4000, overlap=0, scale=1.0)
        for s in scenes:
            try:
                parse_scene(s, ocr, damage_level=args.damage, out_name=parsed_name)
            except Exception as exc:                     # a scene that cannot be parsed counts as all missed
                print(f"{s.name}: parse failed ({exc})")
    results = [score_scene(s, parsed_name) for s in scenes if (s / parsed_name).exists()]
    for r in results:
        bad = len(r["misses"]) + len(r["extras"]) + len(r["errors"])
        print(f"{r['scene']}: {bad} problem(s)" + "".join(f"\n   - {x}" for x in
              ([f"missed {m}" for m in r['misses']] + [f"spurious {e}" for e in r['extras']] + r['errors'])[:12]))
    report = summarise(results)
    print("\n" + report)
    (args.data / f"parser_eval{tag}.md").write_text(report + "\n")
    (args.data / f"parser_eval{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
