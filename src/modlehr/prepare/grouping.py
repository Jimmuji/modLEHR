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
