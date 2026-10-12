import json

import pytest

from ferrosight.datagen.defects import inject_defects
from ferrosight.datagen.drawings import render_drawings
from ferrosight.datagen.structure import generate_design


@pytest.mark.parametrize("seed", range(25))
def test_design_and_defects_are_consistent(seed):
    design = generate_design(seed)
    assert generate_design(seed).to_dict() == design.to_dict()          # deterministic
    ids = [c.id for c in design.components]
    assert len(ids) == len(set(ids))
    assert all(c.length() > 0.02 for c in design.components)

    built, defects = inject_defects(design, seed)
    built_ids = {c.id for c in built}
    targets = [d["component_id"] for d in defects]
    assert len(targets) == len(set(targets))                             # one defect per component
    for d in defects:
        if d["type"] == "missing":
            assert d["component_id"] not in built_ids
        elif d["type"] == "extra":
            assert d["component_id"] in built_ids and d["component_id"] not in ids
        else:
            assert d["component_id"] in built_ids
    untouched = set(ids) - set(targets)
    by_built = {c.id: c for c in built}
    assert all(by_built[i] == design.by_id()[i] for i in untouched)      # nothing else changed


def test_forced_clean_scene():
    design = generate_design(1)
    built, defects = inject_defects(design, 1, n_defects=0)
    assert defects == [] and built == design.components


def test_drawing_annotations_cover_design(tmp_path):
    design = generate_design(4)
    render_drawings(design, tmp_path)
    meta = json.loads((tmp_path / "drawings" / "annotations.json").read_text())
    w, h = meta["sheet_px"]
    annotated = {a["component_id"] for a in meta["annotations"]}
    assert {c.id for c in design.components if c.kind != "baseplate"} <= annotated
    for a in meta["annotations"]:
        x0, y0, x1, y1 = a["tag_bbox_px"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    assert (tmp_path / "drawings.pdf").stat().st_size > 10_000


def test_ifc_roundtrip(tmp_path):
    ifcopenshell = pytest.importorskip("ifcopenshell")
    from ferrosight.datagen.ifc_export import export_ifc
    design = generate_design(2)
    export_ifc(design, tmp_path / "m.ifc")
    model = ifcopenshell.open(str(tmp_path / "m.ifc"))
    assert {e.Name for e in model.by_type("IfcElement")} == {c.id for c in design.components}
