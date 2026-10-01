"""Project binary labels derived from HAM10000 `dx` codes.

HAM10000 is originally a 7-class diagnostic dataset. It does not provide an official
benign/malignant binary target. The mapping below is this project's modeling choice.

    0 Benign:     nv, bkl, df, vasc
    1 Malignant:  mel, bcc, akiec

`akiec` is a single HAM10000 class that combines actinic keratoses and intraepithelial
carcinoma (Bowen's disease). It is grouped into the positive/malignant class for this
binary experiment. That does not mean every actinic keratosis is clinically equivalent
to invasive malignancy.
"""

from __future__ import annotations

CLASS_INDEX_TO_NAME = {
    0: "Benign",
    1: "Malignant",
}

DX_TO_LABEL = {
    "nv": 0,
    "bkl": 0,
    "df": 0,
    "vasc": 0,
    "mel": 1,
    "bcc": 1,
    "akiec": 1,
}

DX_TO_NAME = {
    "nv": "melanocytic nevus",
    "mel": "melanoma",
    "bkl": "benign keratosis-like lesion",
    "bcc": "basal cell carcinoma",
    "akiec": "actinic keratosis / intraepithelial carcinoma (Bowen's)",
    "vasc": "vascular lesion",
    "df": "dermatofibroma",
}


def dx_to_label(dx: str) -> int:
    key = str(dx).strip().lower()
    if key not in DX_TO_LABEL:
        raise ValueError(f"Unmapped HAM10000 dx={dx!r}. Refusing to invent a label.")
    return DX_TO_LABEL[key]
