"""Text reading for drawing sheets.

Drawing text is small relative to the sheet, so the sheet is read in overlapping tiles
(optionally upscaled) and the results are merged. Every result keeps its quadrilateral so
later stages know where on the sheet a string came from.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class TextBox:
    text: str
    conf: float
    quad: list            # 4 points [[x, y], ...] in sheet pixels
    cx: float = 0.0
    cy: float = 0.0
    length: float = 0.0   # extent along the reading direction
    height: float = 0.0   # extent across the reading direction
    angle: float = 0.0    # reading direction in degrees, image coords, folded into (-90, 90]
    bbox: list = field(default_factory=list)

    def __post_init__(self):
        q = np.asarray(self.quad, float)
        self.cx, self.cy = float(q[:, 0].mean()), float(q[:, 1].mean())
        e1, e2 = q[1] - q[0], q[2] - q[1]
        l1, l2 = float(np.linalg.norm(e1)), float(np.linalg.norm(e2))
        long_edge = e1 if l1 >= l2 else e2
        self.length, self.height = max(l1, l2), min(l1, l2)
        a = math.degrees(math.atan2(long_edge[1], long_edge[0]))
        a = a - 180 if a > 90 else a + 180 if a <= -90 else a
        self.angle = a
        self.bbox = [float(q[:, 0].min()), float(q[:, 1].min()), float(q[:, 0].max()), float(q[:, 1].max())]

    @property
    def orientation(self) -> str:
        a = abs(self.angle)
        if len(self.text) <= 2 and self.length < 1.6 * self.height:
            return "h"                      # too short to tell; assume upright
        return "h" if a < 20 else "v" if a > 70 else "d"

    def to_dict(self) -> dict:
        return dict(text=self.text, conf=round(self.conf, 3), quad=[[round(x, 1), round(y, 1)] for x, y in self.quad])


def _iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if inter <= 0:
        return 0.0
    return inter / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def thread_budget() -> int:
    """CPU threads this process may use: the job's slot count on a batch cluster."""
    for var in ("FS_THREADS", "NSLOTS", "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        v = os.environ.get(var, "")
        if v.isdigit() and int(v) > 0:
            return int(v)
    return max(1, min(4, os.cpu_count() or 1))


def _limit_threads(n: int) -> None:
    """ONNX Runtime otherwise starts one thread per core on the machine."""
    import onnxruntime as ort
    import rapidocr_onnxruntime.utils as ru

    def options():
        o = ort.SessionOptions()
        o.intra_op_num_threads = n
        o.inter_op_num_threads = 1
        return o
    ru.SessionOptions = options
    cv2.setNumThreads(n)


class SheetOCR:
    def __init__(self, tile: int = 1100, overlap: int = 260, scale: float = 2.0, threads: int | None = None):
        self.threads = threads or thread_budget()
        _limit_threads(self.threads)
        from rapidocr_onnxruntime import RapidOCR
        self.engine = RapidOCR()
        self.tile, self.overlap, self.scale = tile, overlap, scale

    def _run(self, img):
        res, _ = self.engine(img)
        return res or []

    def read(self, image: np.ndarray) -> list[TextBox]:
        """Read all text on a sheet. `image` is BGR or grayscale, full resolution."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        H, W = image.shape[:2]
        step = self.tile - self.overlap
        boxes: list[TextBox] = []
        for y0 in range(0, max(1, H - self.overlap), step):
            for x0 in range(0, max(1, W - self.overlap), step):
                crop = image[y0:y0 + self.tile, x0:x0 + self.tile]
                if crop.size == 0 or crop.min() > 200:
                    continue                                # blank tile
                ch, cw = crop.shape[:2]
                if self.scale != 1.0:
                    crop = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
                for quad, text, conf in self._run(crop):
                    q = np.asarray(quad, float) / self.scale
                    # drop detections cut off by an interior tile edge; the neighbouring tile has them whole
                    m = 6
                    if ((x0 > 0 and q[:, 0].min() < m) or (y0 > 0 and q[:, 1].min() < m) or
                            (x0 + cw < W and q[:, 0].max() > cw - m) or (y0 + ch < H and q[:, 1].max() > ch - m)):
                        continue
                    q += [x0, y0]
                    text = text.strip()
                    if text:
                        boxes.append(TextBox(text, float(conf), q.tolist()))
        return self._dedupe(boxes)

    @staticmethod
    def _dedupe(boxes: list[TextBox]) -> list[TextBox]:
        boxes = sorted(boxes, key=lambda b: (-len(b.text), -b.conf))
        kept: list[TextBox] = []
        for b in boxes:
            if all(_iou(b.bbox, k.bbox) < 0.5 for k in kept):
                kept.append(b)
        return kept

    def read_crop(self, image: np.ndarray, scale: float = 3.0) -> tuple[str, float]:
        """Recognise a single short string (e.g. a grid bubble label) in a tight crop."""
        if image.size == 0:
            return "", 0.0
        crop = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        res, _ = self.engine(crop, use_det=False, use_cls=False, use_rec=True)
        if not res:
            return "", 0.0
        return str(res[0][0]).strip(), float(res[0][1])
