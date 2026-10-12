"""Steel section catalogue (dimensions in metres)."""

SECTIONS = {
    # wide-flange columns
    "W8x31":  dict(shape="I", d=0.203, bf=0.203, tw=0.0072, tf=0.0110),
    "W10x49": dict(shape="I", d=0.253, bf=0.254, tw=0.0086, tf=0.0142),
    "W12x65": dict(shape="I", d=0.308, bf=0.305, tw=0.0099, tf=0.0154),
    "W14x90": dict(shape="I", d=0.356, bf=0.368, tw=0.0112, tf=0.0180),
    # wide-flange beams
    "W12x26": dict(shape="I", d=0.310, bf=0.165, tw=0.0058, tf=0.0097),
    "W16x40": dict(shape="I", d=0.407, bf=0.178, tw=0.0077, tf=0.0128),
    "W18x50": dict(shape="I", d=0.457, bf=0.190, tw=0.0090, tf=0.0145),
    "W21x62": dict(shape="I", d=0.533, bf=0.209, tw=0.0102, tf=0.0156),
    "W24x76": dict(shape="I", d=0.607, bf=0.228, tw=0.0112, tf=0.0173),
    # hollow sections (braces)
    "HSS4x4x1/4": dict(shape="RECT", d=0.102, bf=0.102, t=0.0064),
    "HSS6x6x3/8": dict(shape="RECT", d=0.152, bf=0.152, t=0.0095),
    "HSS8x8x1/2": dict(shape="RECT", d=0.203, bf=0.203, t=0.0127),
    # base plate (extruded vertically by its thickness)
    "PL500x500x30": dict(shape="RECT", d=0.500, bf=0.500, t=0.030),
}

COLUMN_SECTIONS = ["W10x49", "W12x65", "W14x90"]
GIRDER_SECTIONS = ["W18x50", "W21x62", "W24x76"]
BEAM_SECTIONS = ["W16x40", "W18x50", "W21x62"]
JOIST_SECTIONS = ["W12x26", "W16x40"]
BRACE_SECTIONS = ["HSS6x6x3/8", "HSS8x8x1/2"]
BASEPLATE_SECTION = "PL500x500x30"
BASEPLATE_THICKNESS = 0.030


def depth(name: str) -> float:
    return SECTIONS[name]["d"]


def visibly_different(name: str, min_ratio: float = 1.45) -> list[str]:
    """Sections a person could tell apart from `name` on site: a different
    shape family, or the same family with a clearly different depth."""
    src = SECTIONS[name]
    out = []
    for other, s in SECTIONS.items():
        if other == name or other == BASEPLATE_SECTION:
            continue
        if s["shape"] != src["shape"]:
            if 0.4 < s["d"] / src["d"] < 2.5:
                out.append(other)
            continue
        r = s["d"] / src["d"]
        if r >= min_ratio or r <= 1.0 / min_ratio:
            out.append(other)
    return out
