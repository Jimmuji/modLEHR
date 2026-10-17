"""Target grouping: maps a disease code to the outcome the model predicts.

Inputs can be as fine as the source records; prediction targets should stay a manageable set
(Delphi predicts about 1,300 ICD-10 three-character codes). A grouping is a function from a
MEDS code to a target name, registered under kind "target_grouping". A grouping that needs files
(a mapping table) is registered as a class; data.target_grouping_options are passed to it once.
A grouping may return None to leave a code out of the targets.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path

from modlehr import registry
from modlehr.registry import register

_BUILT: dict[str, object] = {}


def make_grouping(name: str, options: dict | None = None):
    """The grouping as a callable code -> target name (or None); classes are built once per options."""
    options = options or {}
    key = name + json.dumps(options, sort_keys=True)
    if key not in _BUILT:
        obj = registry.get("target_grouping", name)
        if isinstance(obj, type):
            _BUILT[key] = obj(**options)
        elif options:
            raise ValueError(f"target grouping '{name}' takes no options, got {sorted(options)}")
        else:
            _BUILT[key] = obj
    return _BUILT[key]


def _open(path: str | Path):
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"mapping file {path} not found")
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, newline="")


@register("target_grouping", "identity")
def identity(code: str) -> str:
    return code


@register("target_grouping", "icd10_3char")
def icd10_3char(code: str) -> str:
    """ICD10CM//E11.9 or ICD10//E119 → ICD10//E11. Other codes are left unchanged."""
    vocab, sep, rest = code.partition("//")
    if sep and vocab.upper().startswith("ICD10") and rest:
        return "ICD10//" + rest.replace(".", "")[:3].upper()
    return code


@register("target_grouping", "icd_3char")
def icd_3char(code: str) -> str:
    """ICD-9 and ICD-10 to their categories, for sources that mix them (MIMIC-IV).

    DIAGNOSIS//ICD//10//E119 → ICD10//E11; DIAGNOSIS//ICD//9//4280 → ICD9//428 (E codes keep
    four characters, E8889 → ICD9//E888). ICD-9 categories stay a separate family until a
    mapping to ICD-10 (for example the CMS GEMs) is added. Other codes go through icd10_3char.
    """
    parts = code.split("//")
    if len(parts) >= 4 and parts[1] == "ICD" and parts[2] in ("9", "10"):
        version, value = parts[2], parts[3].replace(".", "").upper()
        if version == "10":
            return "ICD10//" + value[:3]
        return "ICD9//" + (value[:4] if value.startswith("E") else value[:3])
    return icd10_3char(code)



def load_gems(path: str | Path) -> dict[str, str]:
    """ICD-9-CM code -> ICD-10 three-character category, from a CMS General Equivalence Mapping
    file (2018_I9gem.txt: source, target and five flag digits per line, the second flag 1 for no
    map). An ICD-9 code with several targets takes the category most of them share, ties broken
    alphabetically."""
    votes: dict[str, Counter] = {}
    with _open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 3 or parts[2][1:2] == "1" or parts[1] == "NoDx":
                continue
            votes.setdefault(parts[0].upper(), Counter())[parts[1][:3].upper()] += 1
    return {icd9: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] for icd9, c in votes.items()}


@register("target_grouping", "icd_gems")
class ICDWithGEMs:
    """ICD-9 and ICD-10 mixed (MIMIC-IV) → ICD-10 three-character categories, with ICD-9 codes mapped
    through the CMS GEMs, so both coding eras land in one target set.

    Options: gems_path, the CMS ICD-9-CM to ICD-10-CM mapping (2018_I9gem.txt). ICD-9 codes the GEMs
    cannot map stay ICD-9 categories (as in icd_3char).
    """

    def __init__(self, gems_path: str):
        self.map = load_gems(gems_path)

    def __call__(self, code: str) -> str | None:
        parts = code.split("//")
        if len(parts) >= 4 and parts[1] == "ICD" and parts[2] == "9":
            target = self.map.get(parts[3].replace(".", "").upper())
            if target:
                return "ICD10//" + target
        return icd_3char(code)


def load_phecode_map(path: str | Path, version: str | None = None) -> dict[str, str]:
    """ICD code (dots removed) -> phecode, from a PheWAS catalog map. The code column is the first
    whose name starts with "icd", the phecode column the one named "phecode" (any case).

    The combined v1.2 map (Phecode_map_v1_2_icd9_icd10cm.csv) holds both ICD versions, told apart
    by its "Flag" column (9 or 10); with `version`, only that version's rows are read, since a few
    codes exist in both with different meanings (ICD-9 E858 is a drug poisoning, ICD-10-CM E858 an
    amyloidosis). A code listed with several phecodes (7,821 ICD-10-CM codes in that file) takes the
    first one listed."""
    with _open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        code_col = next((c for c in fields if c.strip().lower().startswith("icd")), None)
        phe_col = next((c for c in fields if c.strip().lower() == "phecode"), None)
        flag_col = next((c for c in fields if c.strip().lower() == "flag"), None)
        if code_col is None or phe_col is None:
            raise ValueError(f"{path}: expected an ICD code column and a 'phecode' column, found {fields}")
        table: dict[str, str] = {}
        for row in reader:
            if not row[code_col] or not row[phe_col]:
                continue
            if version and flag_col and row[flag_col].strip() != version:
                continue
            table.setdefault(row[code_col].replace(".", "").strip().upper(), row[phe_col].strip())
        return table


@register("target_grouping", "phecode")
class Phecode:
    """ICD-9-CM and ICD-10-CM codes → phecodes, clinically meaningful groups built for EHR research.

    Options: icd9_map and icd10_map, PheWAS catalog map files (either may be left out; the combined
    v1.2 file can be given as both); unmapped: "keep" (default, the code goes through icd_3char) or
    "drop" (not a target). A code not in the map is looked up by ever shorter prefixes, so a more
    specific code finds its parent's phecode.
    """

    def __init__(self, icd9_map: str | None = None, icd10_map: str | None = None, unmapped: str = "keep"):
        if unmapped not in ("keep", "drop"):
            raise ValueError("unmapped must be keep or drop")
        self.maps = {"9": load_phecode_map(icd9_map, "9") if icd9_map else {},
                     "10": load_phecode_map(icd10_map, "10") if icd10_map else {}}
        self.unmapped = unmapped

    def __call__(self, code: str) -> str | None:
        parts = code.split("//")
        if len(parts) >= 4 and parts[1] == "ICD" and parts[2] in self.maps:
            value, table = parts[3].replace(".", "").upper(), self.maps[parts[2]]
            for end in range(len(value), 2, -1):
                if value[:end] in table:
                    return "PHECODE//" + table[value[:end]]
            return None if self.unmapped == "drop" else icd_3char(code)
        return code
