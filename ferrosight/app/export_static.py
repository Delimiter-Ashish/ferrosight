"""Export the FerroSight demo as a static website (GitHub Pages; no server, no GPU).

    python -m ferrosight.app.export_static --data data/synth --out docs/demo

Every flagged and review member gets its images; a few built-as-drawn members per building are
included as examples. Member geometry feeds the 3D view; the VLM's answers (if a vlm_*.json
benchmark exists) are attached for comparison.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from .server import STATIC, create_app


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9-]", "_", s)


def load_vlm(data: Path) -> dict:
    files = sorted(data.glob("vlm_*.json"), key=lambda p: ("7B" not in p.name, p.name))
    if not files:
        return {}
    d = json.loads(files[0].read_text())
    model = d["model"].split("/")[-1]
    return {(r["scene"], r["id"]): dict(model=model, present=r.get("present"), size=r.get("size"), reason=r.get("reason"))
            for r in d["records"] if "present" in r}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/synth"))
    ap.add_argument("--out", type=Path, default=Path("docs/demo"))
    ap.add_argument("--passed", type=int, default=3)
    ap.add_argument("--frames", type=int, default=3)
    args = ap.parse_args()
    out = args.out
    (out / "data").mkdir(parents=True, exist_ok=True)
    client = TestClient(create_app(args.data))
    rng = random.Random(0)
    vlm = load_vlm(args.data)
    scenes = client.get("/api/scenes").json()
    summary = dict(buildings=len(scenes), seen=0, caught=0, correct=0, false_alarms=0)
    want = {"missing": "flag: absent", "misplaced": "flag: absent", "wrong_section": "flag: wrong size"}
    total = 0
    for s in scenes:
        name = s["name"]
        sc = client.get(f"/api/scenes/{name}").json()
        comps = {c["id"]: c for c in json.loads((args.data / name / "parsed_design.json").read_text())["components"]}
        labels = json.loads((args.data / name / "labels.json").read_text())["components"]
        passed = [m for m in sc["members"] if m["outcome"] == "passed" and m["frames"]]
        keep = {m["id"] for m in sc["members"] if m["outcome"] not in ("passed", "not visible")}
        keep |= {m["id"] for m in rng.sample(passed, min(args.passed, len(passed)))}
        img_dir = out / "img" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        for m in sc["members"]:
            c = comps[m["id"]]
            evs = [e for e in c.get("evidence", []) if "line_px" in e or "tag_bbox_px" in e]
            ev = next((e for e in evs if e["sheet"].startswith("S-1")), evs[0] if evs else {})
            lab = labels.get(m["id"], {})
            m.update(start=[round(v, 3) for v in c["start"]], end=[round(v, 3) for v in c["end"]], sheet=ev.get("sheet"),
                     mark_read=c.get("mark_read"), vlm=vlm.get((name, m["id"])))
            if lab.get("status") == "ok":
                summary["correct"] += 1
                summary["false_alarms"] += m["outcome"].startswith("flag")
            elif lab.get("observable"):
                summary["seen"] += 1
                summary["caught"] += m["outcome"] == want[lab["status"]]
            if m["id"] not in keep:
                m.update(images=False, has_drawing=False, frames=[])
                continue
            if m["has_drawing"]:
                r = client.get(f"/api/scenes/{name}/drawing/{m['id']}.png")
                if r.status_code == 200:
                    (img_dir / f"d_{slug(m['id'])}.png").write_bytes(r.content)
                    total += len(r.content)
                else:
                    m["has_drawing"] = False
            good = []
            for f in m["frames"][:args.frames]:
                r = client.get(f"/api/scenes/{name}/frame/{f}/{m['id']}.jpg")
                if r.status_code == 200:
                    (img_dir / f"f_{f}_{slug(m['id'])}.jpg").write_bytes(r.content)
                    total += len(r.content)
                    good.append(f)
            m["frames"] = good
        sc["has_video"] = False
        (out / "data" / f"{name}.json").write_text(json.dumps(sc, separators=(",", ":")))
        print(f"{name}: {len(keep)} members with images", flush=True)
    (out / "data" / "scenes.json").write_text(json.dumps(scenes))
    (out / "data" / "summary.json").write_text(json.dumps(summary))
    shutil.copy(STATIC / "demo.html", out / "index.html")
    (out.parent / ".nojekyll").write_text("")
    print(f"\nsummary: {summary}\nwrote {out}/index.html and {total / 1e6:.1f} MB of images")


if __name__ == "__main__":
    main()
