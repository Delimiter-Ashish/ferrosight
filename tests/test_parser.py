import json

import cv2
import numpy as np
import pytest

from ferrosight.datagen.drawings import render_drawings
from ferrosight.datagen.structure import generate_design
from ferrosight.parser.preprocess import DAMAGE, damage, prepare
from ferrosight.parser.raster import extract
from ferrosight.parser.text import match_section, split_tag, valid_mark


@pytest.mark.parametrize("raw, section, mark", [
    ("W2 1×76 [G-A12- .2]", "W24x76", "G-A12-.2"),          # joist line erased through the tag
    ("PL500×50030 [BP-B1]", "PL500x500x30", "BP-B1"),      # dropped separator
    ("VB HSS8×8×1/2 [BR-A12-L2]", "HSS8x8x1/2", "BR-A12-L2"),
    ("2 IBR-3CD-L2Z]", None, "BR-3CD-L2Z"),                 # lost opening bracket
    ("W10×4g", "W10x49", None),
])
def test_tag_reading(raw, section, mark):
    head, m = split_tag(raw)
    assert match_section(head)[0] == section
    assert m == mark


def test_mark_grammar():
    for ok in ["G-A12-L1", "B-1AB-L3", "J-12.1AB-L2", "BR-A12-L1", "BR-3BC-L2", "C-D4-L1", "BP-A1"]:
        assert valid_mark(ok)
    for bad in ["G-A12-L", "BR-3CD-L2Z", "J-12AB-L2", "C-A1"]:
        assert not valid_mark(bad)


@pytest.fixture(scope="module")
def sheets(tmp_path_factory):
    out = tmp_path_factory.mktemp("sheets")
    design = generate_design(3)
    render_drawings(design, out)
    return design, out


def test_plan_linework_matches_design(sheets):
    design, out = sheets
    gray = cv2.imread(str(out / "drawings" / "S-102.png"), cv2.IMREAD_GRAYSCALE)
    lw = extract(gray)
    lvl = [c for c in design.components if c.level == 2]
    count = lambda k: sum(c.kind == k for c in lvl)
    assert len(lw.squares) == count("column")
    solid_h = [s for s in lw.horizontal if not s.dashed and s.length < 0.5 * gray.shape[1]]
    assert sum(s.dashed for s in lw.horizontal + lw.vertical) == count("brace")
    # girders are the solid horizontal lines inside the drawing; the border and title block are longer or outside
    assert len([s for s in solid_h if s.thickness > 3.5]) >= count("girder")


def test_elevation_finds_diagonals(sheets):
    design, out = sheets
    ann = json.loads((out / "drawings" / "annotations.json").read_text())["annotations"]
    braces = [a for a in ann if a["sheet"] == "S-201" and a["kind"] == "brace"]
    lw = extract(cv2.imread(str(out / "drawings" / "S-201.png"), cv2.IMREAD_GRAYSCALE))
    assert len(lw.diagonal) == len(braces)


@pytest.mark.parametrize("seed", [2, 3, 7])
def test_deskew_recovers_rotation(sheets, seed):
    _, out = sheets
    gray = cv2.imread(str(out / "drawings" / "S-101.png"), cv2.IMREAD_GRAYSCALE)
    applied = float(np.random.default_rng(seed).uniform(-DAMAGE["heavy"]["rot"], DAMAGE["heavy"]["rot"]))
    _, info = prepare(damage(gray, "heavy", seed))
    assert abs(info["skew_deg"] - applied) < 0.05 or abs(applied) < 0.03


def test_full_parse_matches_design(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    from ferrosight.datagen.generate import build_scene
    from ferrosight.parser.assemble import parse_scene
    from ferrosight.parser.evaluate import score_scene
    scene = build_scene(2, tmp_path)
    parse_scene(scene)
    r = score_scene(scene)
    for kind, s in r["per_kind"].items():
        assert s["matched"] == s["gt"] == s["parsed"], kind
        assert s["section_ok"] == s["matched"], kind
    assert r["grid_err"] < 0.01 and r["level_err"] < 0.01
