"""Task definitions and label protocols.

CheXpert uses the standard U-Ignore / U-Ones / U-Zeros uncertain-label
protocol. VinDr-CXR / VinBigData has no Pneumonia class; pneumonia_proxy
on the target domain is Consolidation ∪ Infiltration and is therefore a
label-ontology shift, not a pure population shift.
"""

from __future__ import annotations

from dataclasses import dataclass

CHEXPERT_FINDING_COLS = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

VINDR_CLASS_ID = {
    "Aortic enlargement": 0,
    "Atelectasis": 1,
    "Calcification": 2,
    "Cardiomegaly": 3,
    "Consolidation": 4,
    "ILD": 5,
    "Infiltration": 6,
    "Lung Opacity": 7,
    "Nodule/Mass": 8,
    "Other lesion": 9,
    "Pleural effusion": 10,
    "Pleural thickening": 11,
    "Pneumothorax": 12,
    "Pulmonary fibrosis": 13,
    "No finding": 14,
}

VINDR_PNEUMONIA_PROXY = ("Consolidation", "Infiltration")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    chexpert_columns: tuple[str, ...]
    vindr_classes: tuple[str, ...]
    positive_is_no_finding: bool = False
    ontology_shift: bool = False
    description: str = ""


TASKS: dict[str, TaskSpec] = {
    "no_finding": TaskSpec(
        name="no_finding",
        chexpert_columns=("No Finding",),
        vindr_classes=("No finding",),
        positive_is_no_finding=True,
        description="No Finding vs any abnormality. Cleanest cross-dataset ontology.",
    ),
    "cardiomegaly": TaskSpec(
        name="cardiomegaly",
        chexpert_columns=("Cardiomegaly",),
        vindr_classes=("Cardiomegaly",),
        description="Overlapping Cardiomegaly label.",
    ),
    "pleural_effusion": TaskSpec(
        name="pleural_effusion",
        chexpert_columns=("Pleural Effusion",),
        vindr_classes=("Pleural effusion",),
        description="Overlapping pleural effusion label.",
    ),
    "consolidation": TaskSpec(
        name="consolidation",
        chexpert_columns=("Consolidation",),
        vindr_classes=("Consolidation",),
        description="Overlapping consolidation label.",
    ),
    "pneumonia_proxy": TaskSpec(
        name="pneumonia_proxy",
        chexpert_columns=("Pneumonia",),
        vindr_classes=VINDR_PNEUMONIA_PROXY,
        ontology_shift=True,
        description=(
            "CheXpert Pneumonia vs VinDr Consolidation∪Infiltration. "
            "Sensitivity analysis only; this is label-ontology shift."
        ),
    ),
}


def parse_chexpert_cell(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return float(text)


def chexpert_binary(value, u_protocol: str = "u_ignore") -> int | None:
    """Map a CheXpert cell to {0, 1} or None (drop).

    Blank / not-mentioned is treated as negative (0), matching the
    CheXpert competition convention. Uncertain (-1) follows u_protocol.
    """
    parsed = parse_chexpert_cell(value)
    if parsed is None:
        return 0
    if parsed == 1.0:
        return 1
    if parsed == 0.0:
        return 0
    if parsed == -1.0:
        protocol = u_protocol.lower()
        if protocol == "u_ignore":
            return None
        if protocol == "u_ones":
            return 1
        if protocol == "u_zeros":
            return 0
        raise ValueError(f"Unknown u_protocol: {u_protocol}")
    return 0


def chexpert_task_label(row: dict, spec: TaskSpec, u_protocol: str) -> int | None:
    if spec.positive_is_no_finding:
        return chexpert_binary(row.get("No Finding"), u_protocol="u_ignore")
    votes = [chexpert_binary(row.get(col), u_protocol) for col in spec.chexpert_columns]
    if any(v is None for v in votes):
        return None
    return int(any(v == 1 for v in votes))


def vindr_task_label(classes: set[str], spec: TaskSpec) -> int:
    if spec.positive_is_no_finding:
        abnormal = {c for c in classes if c != "No finding"}
        return int(len(abnormal) == 0 and "No finding" in classes)
    return int(any(c in classes for c in spec.vindr_classes))
