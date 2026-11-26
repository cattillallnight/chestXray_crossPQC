"""Cohort construction for CheXpert (source) and VinDr-CXR (target)."""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from qcalibnet.labels import TASKS, TaskSpec, chexpert_task_label, vindr_task_label
from qcalibnet.paths import (
    CHEXPERT_TRAIN_CSV,
    CHEXPERT_VALID_CSV,
    DATA_DIR,
    VINDR_META_CSV,
    chexpert_image_path,
    ensure_dirs,
    resolve_vindr_label_csv,
    vindr_image_path,
)


def _is_real_image(path: Path) -> bool:
    if path.name.startswith("._"):
        return False
    try:
        return os.stat(path).st_size > 1024
    except OSError:
        return False


def load_chexpert_table(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["patient_id"] = df["Path"].str.extract(r"(patient\d+)", expand=False)
    df["image_path"] = df["Path"].map(lambda p: str(chexpert_image_path(str(p))))
    cache_path = DATA_DIR / f"{csv_path.stem}_exists_cache.csv"
    if cache_path.is_file():
        cached = pd.read_csv(cache_path)
        df = df.merge(cached, on="image_path", how="left")
        missing = df["exists"].isna()
        if missing.any():
            df.loc[missing, "exists"] = [
                _is_real_image(Path(p)) for p in df.loc[missing, "image_path"]
            ]
            ensure_dirs()
            df[["image_path", "exists"]].drop_duplicates("image_path").to_csv(cache_path, index=False)
        df["exists"] = df["exists"].astype(bool)
        return df
    df["exists"] = [_is_real_image(Path(p)) for p in df["image_path"]]
    ensure_dirs()
    df[["image_path", "exists"]].drop_duplicates("image_path").to_csv(cache_path, index=False)
    return df


def build_chexpert_cohort(
    spec: TaskSpec,
    u_protocol: str = "u_ignore",
    frontal_only: bool = True,
    include_official_valid: bool = False,
) -> pd.DataFrame:
    frames = [load_chexpert_table(CHEXPERT_TRAIN_CSV)]
    if include_official_valid:
        frames.append(load_chexpert_table(CHEXPERT_VALID_CSV))
    df = pd.concat(frames, ignore_index=True)
    if frontal_only:
        df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df = df[df["exists"]].copy()
    labels = [chexpert_task_label(row, spec, u_protocol) for row in df.to_dict("records")]
    df["label"] = labels
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)
    df["domain"] = "chexpert"
    df["task"] = spec.name
    df["u_protocol"] = u_protocol
    keep = [
        "Path",
        "patient_id",
        "image_path",
        "Sex",
        "Age",
        "Frontal/Lateral",
        "AP/PA",
        "label",
        "domain",
        "task",
        "u_protocol",
    ]
    return df[keep].reset_index(drop=True)


def _stratified_split(df: pd.DataFrame, test_size: float, seed: int, col: str):
    counts = df[col].value_counts()
    can_stratify = df[col].nunique() >= 2 and int(counts.min()) >= 2
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df[col] if can_stratify else None,
        )
    except ValueError:
        return train_test_split(df, test_size=test_size, random_state=seed)


def split_chexpert_patient_level(
    cohort: pd.DataFrame,
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    balanced_train: bool = True,
) -> dict[str, pd.DataFrame]:
    """Patient-level split. Train may be class-balanced; val/test keep natural prevalence unless undersampled to requested n."""
    patients = (
        cohort.groupby("patient_id")["label"]
        .max()
        .reset_index()
        .rename(columns={"label": "patient_label"})
    )
    train_p, rest_p = _stratified_split(patients, test_size=0.40, seed=seed, col="patient_label")
    val_p, test_p = _stratified_split(rest_p, test_size=0.50, seed=seed, col="patient_label")

    def take_images(patient_ids, n, balanced: bool) -> pd.DataFrame:
        subset = cohort[cohort["patient_id"].isin(set(patient_ids))].copy()
        subset = subset.drop_duplicates("image_path")
        if balanced:
            pos = subset[subset["label"] == 1]
            neg = subset[subset["label"] == 0]
            half = n // 2
            n_pos = min(len(pos), half)
            n_neg = min(len(neg), n - n_pos)
            if n_pos + n_neg < n:
                leftover = n - n_pos - n_neg
                extra_pos = min(len(pos) - n_pos, leftover)
                n_pos += extra_pos
                leftover -= extra_pos
                n_neg += min(len(neg) - n_neg, leftover)
            pos_s = pos.sample(n=n_pos, random_state=seed) if n_pos else pos
            neg_s = neg.sample(n=n_neg, random_state=seed) if n_neg else neg
            out = pd.concat([pos_s, neg_s], ignore_index=True)
        else:
            n_take = min(n, len(subset))
            out = subset.sample(n=n_take, random_state=seed)
        return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    splits = {
        "train": take_images(train_p["patient_id"], n_train, balanced=balanced_train),
        "val": take_images(val_p["patient_id"], n_val, balanced=False),
        "test": take_images(test_p["patient_id"], n_test, balanced=False),
    }
    for name, frame in splits.items():
        frame["split"] = name
    return splits


def load_vindr_annotations() -> pd.DataFrame:
    label_csv = resolve_vindr_label_csv()
    if label_csv is None:
        raise FileNotFoundError(
            "VinDr-CXR annotations were not found. Set QCALIBNET_VINDR_DIR to the "
            "extracted VinDr-CXR directory or place vindr_train.csv under "
            "QCALIBNET_DATA_DIR/external. Required columns: image_id, class_name "
            "(and class_id if present)."
        )
    df = pd.read_csv(label_csv)
    required = {"image_id", "class_name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label_csv} missing columns {missing}")
    return df


def build_vindr_cohort(spec: TaskSpec, vote: str = "any") -> pd.DataFrame:
    ann = load_vindr_annotations()
    grouped: dict[str, set[str]] = defaultdict(set)
    for row in ann.itertuples(index=False):
        grouped[str(row.image_id)].add(str(row.class_name))

    records = []
    for image_id, classes in grouped.items():
        path = vindr_image_path(image_id)
        if not _is_real_image(path):
            continue
        label = vindr_task_label(classes, spec)
        records.append(
            {
                "image_id": image_id,
                "image_path": str(path),
                "label": int(label),
                "n_classes": len(classes),
                "classes": "|".join(sorted(classes)),
                "domain": "vindr",
                "task": spec.name,
                "vote": vote,
            }
        )
    if not records:
        raise RuntimeError("VinDr cohort is empty after joining labels to PNGs.")
    return pd.DataFrame.from_records(records)


def split_vindr_holdout(
    cohort: pd.DataFrame,
    n_test: int,
    seed: int,
    balanced: bool = True,
) -> dict[str, pd.DataFrame]:
    pos = cohort[cohort["label"] == 1]
    neg = cohort[cohort["label"] == 0]
    if balanced and len(pos) and len(neg):
        half = n_test // 2
        n_pos = min(len(pos), half)
        n_neg = min(len(neg), n_test - n_pos)
        test = pd.concat(
            [
                pos.sample(n=n_pos, random_state=seed),
                neg.sample(n=n_neg, random_state=seed),
            ],
            ignore_index=True,
        )
    else:
        n_take = min(n_test, len(cohort))
        test = cohort.sample(n=n_take, random_state=seed)
    test = test.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test["split"] = "external_test"
    remainder = cohort[~cohort["image_id"].isin(set(test["image_id"]))].copy()
    remainder["split"] = "unused"
    return {"external_test": test, "unused": remainder}


def cohort_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    n_pos = int(df["label"].sum()) if n else 0
    out = {
        "n": n,
        "n_pos": n_pos,
        "n_neg": n - n_pos,
        "prevalence": float(n_pos / n) if n else 0.0,
    }
    if "patient_id" in df.columns:
        out["n_patients"] = int(df["patient_id"].nunique())
    return out


def vindr_labels_available() -> bool:
    return resolve_vindr_label_csv() is not None


def vindr_meta_count() -> int:
    if not VINDR_META_CSV.is_file():
        return 0
    return max(sum(1 for _ in open(VINDR_META_CSV, encoding="utf-8")) - 1, 0)
