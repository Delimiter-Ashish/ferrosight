"""Turning OCR strings into structured values: steel sections, member marks, numbers."""
from __future__ import annotations

import re

from ..datagen.sections import SECTIONS

KINDS = {"G": "girder", "B": "beam", "J": "joist", "BR": "brace", "C": "column", "BP": "baseplate"}

# Mark grammar of this drawing set (general note 5 on S-001):
#   G-A12-L1   girder on gridline A between 1 and 2, level 1
#   B-1AB-L1   beam on gridline 1 between A and B
#   J-12.1AB-L1 first joist in bay 1-2, spanning A-B
#   BR-A12-L1 / BR-1AB-L1  brace in that frame bay, storey 1
#   C-A1-L1    column at A/1, storey 1
#   BP-A1      base plate at A/1
MARK_RE = re.compile(
    r"^(?:G-[A-H]\d\d-L\d|B-\d[A-H][A-H]-L\d|J-\d\d\.\d[A-H][A-H]-L\d|"
    r"BR-(?:[A-H]\d\d|\d[A-H][A-H])-L\d|C-[A-H]\d-L\d|BP-[A-H]\d)$")
_LOOSE_MARK = re.compile(r"(BR|BP|[GBJC])\s*-\s*([A-Z0-9.:,]{2,8})(?:\s*-\s*([A-Z0-9:,]{1,3}))?")
_SECTION_NAMES = {re.sub(r"[^A-Z0-9/]", "", k.upper().replace("X", "X")): k for k in SECTIONS}


def normalise(text: str) -> str:
    t = text.upper().replace("×", "X").replace("*", "X")
    t = t.replace("【", "[").replace("】", "]").replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", " ", t).strip()


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _clean_section_text(raw: str) -> str:
    t = normalise(raw)
    if t.startswith("VB"):
        t = t[2:]
    t = re.sub(r"[^A-Z0-9/]", "", t)
    return t[:1] + t[1:].replace("O", "0") if t else t


def match_section(raw: str, prefer: list[str] | set[str] | None = None) -> tuple[str | None, str]:
    """Closest catalogue section to an OCR string. Returns (section, how) where how is
    'exact', 'fuzzy', or 'none'. Each whitespace-separated token is tried as well as the whole
    string, so stray characters next to the section do not count against it. `prefer` (e.g.
    the sections the schedule lists for this member type and level) breaks ties."""
    texts = {_clean_section_text(raw)} | {_clean_section_text(tok) for tok in normalise(raw).split()}
    texts = {t for t in texts if len(t) >= 3}
    if not texts:
        return None, "none"
    for t in texts:
        if t in _SECTION_NAMES:
            return _SECTION_NAMES[t], "exact"
    scored = sorted((edit_distance(t, k), 0 if prefer and v in prefer else 1, v, t)
                    for t in texts for k, v in _SECTION_NAMES.items())
    best_d, _, best, t = scored[0]
    if best_d <= max(1, len(t) // 3):
        return best, "fuzzy"
    return None, "none"


def split_tag(text: str) -> tuple[str, str | None]:
    """Split 'W16x40  [J-12.1AB-L2]' into ('W16X40', 'J-12.1AB-L2'). Tolerates a lost bracket."""
    t = normalise(text)
    if "[" in t:
        head, tail = t.split("[", 1)
        mark = tail.split("]", 1)[0]
        return head.strip(), clean_mark(mark)
    m = _LOOSE_MARK.search(t)
    if m and (m.group(1) in ("BR", "BP") or m.group(3)):
        return t[:m.start()].strip(" I|["), clean_mark(t[m.start():])
    return t, None


def clean_mark(mark: str) -> str | None:
    m = normalise(mark).strip(" []|")
    m = m.replace(" ", "").replace(",", "-").replace(":", "")
    return m or None


def valid_mark(mark: str | None) -> bool:
    return bool(mark) and bool(MARK_RE.match(mark))


def parse_number(text: str) -> float | None:
    t = normalise(text).replace("O", "0").replace(",", ".")
    m = re.search(r"[-+]?\d+\.\d+", t)
    return float(m.group()) if m else None
