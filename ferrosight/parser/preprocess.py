"""Sheet clean-up before parsing (contrast, skew), and synthetic scan damage for robustness tests."""
from __future__ import annotations

import math

import cv2
import numpy as np


def normalise_contrast(gray: np.ndarray) -> np.ndarray:
    """Stretch so paper is white and the darkest ink is black. Faded or grey scans become usable
    with the parser's fixed ink threshold."""
    paper = float(np.percentile(gray, 60))
    ink = float(np.percentile(gray, 0.2))
    if paper - ink < 40:
        return gray
    out = (gray.astype(np.float32) - ink) * (255.0 / (paper - ink))
    return np.clip(out, 0, 255).astype(np.uint8)


def estimate_skew(gray: np.ndarray, max_deg: float = 1.5) -> float:
    """Small rotation (degrees) that makes the sheet's ink rows sharpest: the projection-profile
    method. Drawings are full of long horizontal lines, so the right angle gives a very peaky
    row histogram."""
    small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ink = (small < 128).astype(np.float32)
    h, w = ink.shape

    def sharpness(a):
        m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        r = cv2.warpAffine(ink, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        prof = r.sum(axis=1)
        return float((prof ** 2).sum())

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.1)
    best = coarse[int(np.argmax([sharpness(a) for a in coarse]))]
    fine = np.arange(best - 0.1, best + 0.1 + 1e-9, 0.01)
    return -float(fine[int(np.argmax([sharpness(a) for a in fine]))])


def deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    a = estimate_skew(gray)
    if abs(a) < 0.03:
        return gray, 0.0
    h, w = gray.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -a, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), a


def prepare(gray: np.ndarray) -> tuple[np.ndarray, dict]:
    g, a = deskew(normalise_contrast(gray))
    return g, dict(skew_deg=round(a, 3))


DAMAGE = {
    "light": dict(rot=0.35, blur=0.7, noise=6.0, fade=0.85, jpeg=60, specks=300),
    "heavy": dict(rot=0.9, blur=1.1, noise=14.0, fade=0.65, jpeg=35, specks=1500),
}


def damage(gray: np.ndarray, level: str, seed: int) -> np.ndarray:
    """Imitate a photocopied / scanned sheet: slight rotation, blur, faded ink, noise, specks, JPEG."""
    p = DAMAGE[level]
    rng = np.random.default_rng(seed)
    h, w = gray.shape
    g = gray.astype(np.float32)
    angle = float(rng.uniform(-p["rot"], p["rot"]))
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    g = cv2.warpAffine(g, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
    g = cv2.GaussianBlur(g, (0, 0), p["blur"])
    g = 255 - (255 - g) * p["fade"]                  # ink fades towards the paper colour
    g = g * rng.uniform(0.9, 1.0)                    # paper is not pure white
    g = g + rng.normal(0, p["noise"], g.shape)
    ys, xs = rng.integers(0, h, p["specks"]), rng.integers(0, w, p["specks"])
    g[ys, xs] = rng.uniform(0, 120, p["specks"])
    g = np.clip(g, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, p["jpeg"]])
    return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
