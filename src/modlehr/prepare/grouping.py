"""Target grouping: maps a disease code to the outcome the model predicts.

Inputs can be as fine as the source records; prediction targets should stay a manageable set
(Delphi predicts about 1,300 ICD-10 three-character codes). A grouping is a function from a
MEDS code to a target name, registered under kind "target_grouping".
"""

from __future__ import annotations

from modlehr.registry import register


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

