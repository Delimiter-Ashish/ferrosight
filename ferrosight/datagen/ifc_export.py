"""Export a list of components to an IFC4 model (the BIM version of the design)."""
from __future__ import annotations

from pathlib import Path

import ifcopenshell
import ifcopenshell.api.aggregate
import ifcopenshell.api.context
import ifcopenshell.api.geometry
import ifcopenshell.api.project
import ifcopenshell.api.pset
import ifcopenshell.api.root
import ifcopenshell.api.spatial
import ifcopenshell.api.unit

from .geometry import local_frame, placement_matrix
from .sections import SECTIONS
from .structure import Design

IFC_CLASS = dict(column="IfcColumn", girder="IfcBeam", beam="IfcBeam", joist="IfcBeam",
                 brace="IfcMember", baseplate="IfcPlate")


def export_ifc(design: Design, path: str | Path, components=None) -> None:
    api = ifcopenshell.api
    f = api.project.create_file(version="IFC4")
    project = api.root.create_entity(f, ifc_class="IfcProject", name=design.name)
    api.unit.assign_unit(f, length={"is_metric": True, "raw": "METERS"})
    model = api.context.add_context(f, context_type="Model")
    body = api.context.add_context(f, context_type="Model", context_identifier="Body",
                                   target_view="MODEL_VIEW", parent=model)
    site = api.root.create_entity(f, ifc_class="IfcSite", name="Site")
    building = api.root.create_entity(f, ifc_class="IfcBuilding", name=design.name)
    api.aggregate.assign_object(f, products=[site], relating_object=project)
    api.aggregate.assign_object(f, products=[building], relating_object=site)
    storeys = {}
    for k, z in enumerate([0.0] + design.levels):
        st = api.root.create_entity(f, ifc_class="IfcBuildingStorey", name="Foundation" if k == 0 else f"Level {k}")
        st.Elevation = float(z)
        api.aggregate.assign_object(f, products=[st], relating_object=building)
        storeys[k] = st

    profiles = {}

    def profile(name):
        if name not in profiles:
            s = SECTIONS[name]
            if s["shape"] == "I":
                profiles[name] = f.create_entity(
                    "IfcIShapeProfileDef", ProfileType="AREA", ProfileName=name, OverallWidth=s["bf"],
                    OverallDepth=s["d"], WebThickness=s["tw"], FlangeThickness=s["tf"])
            elif name.startswith("HSS"):
                profiles[name] = f.create_entity(
                    "IfcRectangleHollowProfileDef", ProfileType="AREA", ProfileName=name,
                    XDim=s["bf"], YDim=s["d"], WallThickness=s["t"])
            else:
                profiles[name] = f.create_entity(
                    "IfcRectangleProfileDef", ProfileType="AREA", ProfileName=name, XDim=s["bf"], YDim=s["d"])
        return profiles[name]

    for c in components if components is not None else design.components:
        el = api.root.create_entity(f, ifc_class=IFC_CLASS[c.kind], name=c.id)
        el.ObjectType = c.kind
        el.Description = f"{c.section} @ {c.grid}"
        length = local_frame(c.start, c.end, c.up)[3]
        rep = api.geometry.add_profile_representation(f, context=body, profile=profile(c.section), depth=length)
        api.geometry.assign_representation(f, product=el, representation=rep)
        api.geometry.edit_object_placement(f, product=el, matrix=placement_matrix(c.start, c.end, c.up), is_si=True)
        api.spatial.assign_container(f, products=[el], relating_structure=storeys[c.level])
        pset = api.pset.add_pset(f, product=el, name="FerroSight")
        api.pset.edit_pset(f, pset=pset, properties=dict(Mark=c.id, Section=c.section, Grid=c.grid, Kind=c.kind))
    f.write(str(path))
