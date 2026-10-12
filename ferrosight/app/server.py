"""Phase 6: FerroSight inspection viewer.

    python -m ferrosight.app.server --data data/synth --port 8050

Open it from the VS Code PORTS tab, or at https://<ondemand host>/rnode/<node>/8050/
(keep the trailing slash: every link in the page is relative).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response

from ..matching.match import project, width_of
from ..verify.verify import outcome

STATIC = Path(__file__).with_name("static")
SCENE_RE = re.compile(r"^scene_\d{4}$")
BGR = {"flag: absent": (48, 48, 220), "flag: wrong size": (0, 130, 255), "review": (0, 190, 235),
       "passed": (70, 170, 60), "not visible": (150, 150, 150)}


def png(img) -> Response:
    ok, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), media_type="image/png")


def jpg(img) -> Response:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return Response(buf.tobytes(), media_type="image/jpeg")


def reason(m) -> str:
    o = m["outcome"]
    if o == "flag: absent":
        return f"Spot seen clearly in {m['clear_views']} views; no steel there."
    if o == "flag: wrong size":
        s = m["size"]
        return f"Measures {s['ratio_to_drawn']}x the drawn size; looks like {s['looks_like']}."
    if o == "review":
        if m["size"] and m["size"]["ratio_to_drawn"] >= 1.45:
            return f"Looks {m['size']['ratio_to_drawn']}x thicker than drawn; may be neighbouring steel."
        return "Views disagree; a person should look."
    if o == "not visible":
        return "The camera never had a clear view."
    return "Steel found where the drawings put it."


def create_app(data: Path) -> FastAPI:
    app = FastAPI(title="FerroSight")

    @lru_cache(maxsize=64)
    def load(name: str) -> dict:
        if not SCENE_RE.match(name) or not (data / name / "verify.json").exists():
            raise HTTPException(404, "scene not found or not processed")
        d = data / name
        J = lambda p: json.loads((d / p).read_text())
        design, verify, matching = J("parsed_design.json"), J("verify.json"), J("matching.json")
        labels = J("labels.json")["components"] if (d / "labels.json").exists() else {}
        comps = {c["id"]: c for c in design["components"]}
        by_match = {m["id"]: m for m in matching["members"]}
        members = []
        for r in verify["members"]:
            c, m = comps[r["id"]], by_match.get(r["id"], {})
            frames = [x["frame"] for x in r.get("closeups", [])] or [x["frame"] for x in m.get("evidence", [])]
            item = dict(id=r["id"], kind=r["kind"], section=r["section"], grid=r["grid"], level=c["level"],
                        outcome=outcome(r), match_verdict=r["match_verdict"], size=r["size"],
                        clear_views=m.get("clear_views", 0), steel_ratio=m.get("steel_ratio"),
                        frames=list(dict.fromkeys(frames))[:3], truth=labels.get(r["id"], {}).get("status"),
                        has_drawing=bool(c.get("evidence")))
            item["reason"] = reason(item)
            members.append(item)
        return dict(design=design, cam=J("render/cameras.json"), comps=comps, members=members,
                    by_id={m["id"]: m for m in members})

    def member(name, member_id):
        s = load(name)
        if member_id not in s["comps"]:
            raise HTTPException(404, "unknown member")
        return s, s["comps"][member_id], s["by_id"][member_id]

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/scenes")
    def scenes():
        out = []
        for d in sorted(data.glob("scene_*")):
            if (d / "verify.json").exists():
                cnt = Counter(m["outcome"] for m in load(d.name)["members"])
                out.append(dict(name=d.name, counts=cnt, total=sum(cnt.values())))
        return out

    @app.get("/api/scenes/{name}")
    def scene(name: str):
        s = load(name)
        return dict(name=name, levels=s["design"]["levels"], grid_x=s["design"]["grid_x"],
                    grid_y=s["design"]["grid_y"], members=s["members"],
                    has_video=(data / name / "site_walkthrough.mp4").exists())

    @app.get("/api/scenes/{name}/video")
    def video(name: str):
        load(name)
        p = data / name / "site_walkthrough.mp4"
        if not p.exists():
            raise HTTPException(404)
        return FileResponse(p, media_type="video/mp4")

    @app.get("/api/scenes/{name}/drawing/{member_id}.png")
    def drawing(name: str, member_id: str):
        s, c, m = member(name, member_id)
        evs = sorted(c.get("evidence", []), key=lambda e: (not e["sheet"].startswith("S-1"), "line_px" not in e))
        if not evs:
            raise HTTPException(404, "member not marked on any sheet")
        e = evs[0]
        img = cv2.imread(str(data / name / "drawings" / f"{e['sheet']}.png"))
        boxes = []
        if "line_px" in e:
            x0, y0, x1, y1 = e["line_px"]
            boxes.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
        if "tag_bbox_px" in e:
            boxes.append(e["tag_bbox_px"])
        b = np.array(boxes, float)
        X0, Y0, X1, Y1 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
        col = BGR[m["outcome"]]
        for bx in boxes:
            cv2.rectangle(img, (int(bx[0]) - 8, int(bx[1]) - 8), (int(bx[2]) + 8, int(bx[3]) + 8), col, 3)
        pad = 160
        H, W = img.shape[:2]
        x0, y0 = max(0, int(X0) - pad), max(0, int(Y0) - pad)
        x1, y1 = min(W, int(X1) + pad), min(H, int(Y1) + pad)
        crop = img[y0:y1, x0:x1]
        if crop.shape[1] > 1100:
            crop = cv2.resize(crop, None, fx=1100 / crop.shape[1], fy=1100 / crop.shape[1], interpolation=cv2.INTER_AREA)
        cv2.putText(crop, f"Sheet {e['sheet']}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2, cv2.LINE_AA)
        return png(crop)

    @app.get("/api/scenes/{name}/frame/{frame}/{member_id}.jpg")
    def frame_image(name: str, frame: int, member_id: str):
        s, c, m = member(name, member_id)
        cam = s["cam"]
        if not 0 <= frame < len(cam["frames"]):
            raise HTTPException(404)
        img = cv2.imread(str(data / name / "render" / "frames" / f"frame_{frame:04d}.jpg"))
        H, W = img.shape[:2]
        a, b = np.asarray(c["start"], float), np.asarray(c["end"], float)
        if c["kind"] == "baseplate":
            mid = (a + b) / 2
            pts = mid + np.array([[-.25, -.25, 0], [.25, -.25, 0], [.25, .25, 0], [-.25, .25, 0]])
        else:
            pts = a + np.linspace(0, 1, 24)[:, None] * (b - a)
        u, v, z = project(pts, cam["frames"][frame]["cam_to_world"], cam)
        front = z > 0.1
        if front.sum() < 2:
            raise HTTPException(404, "member behind the camera in this frame")
        P = np.round(np.c_[u[front], v[front]]).astype(np.int32).reshape(-1, 1, 2)
        col = BGR[m["outcome"]]
        t = int(np.clip(cam["fx"] * width_of(c) / max(float(z[front].mean()), 0.1), 3, 70))
        over = img.copy()
        if c["kind"] == "baseplate":
            cv2.fillConvexPoly(over, P, col)
        else:
            cv2.polylines(over, [P], False, col, t, cv2.LINE_AA)
        img = cv2.addWeighted(over, 0.42, img, 0.58, 0)
        cv2.polylines(img, [P], c["kind"] == "baseplate", col, 2, cv2.LINE_AA)
        uu, vv = u[front], v[front]
        m_ = 0.7 * max(np.ptp(uu), np.ptp(vv)) + 60
        x0, y0 = int(max(0, uu.min() - m_)), int(max(0, vv.min() - m_))
        x1, y1 = int(min(W, uu.max() + m_)), int(min(H, vv.max() + m_))
        if x1 - x0 < 320:
            cx = (x0 + x1) // 2
            x0, x1 = max(0, cx - 160), min(W, cx + 160)
        if y1 - y0 < 200:
            cy = (y0 + y1) // 2
            y0, y1 = max(0, cy - 100), min(H, cy + 100)
        crop = img[y0:y1, x0:x1]
        scale = min(2.0, 640 / max(1, crop.shape[1]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        cv2.putText(crop, f"frame {frame}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return jpg(crop)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(description="FerroSight inspection viewer")
    ap.add_argument("--data", type=Path, default=Path("data/synth"))
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    args = ap.parse_args()
    uvicorn.run(create_app(args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
