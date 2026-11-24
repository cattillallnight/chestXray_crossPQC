import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS
from qcalibnet.datasets import build_chexpert_cohort, _stratified_split

print("=== POOL TRAIN CO BAO NHIEU POS/NEG TRUOC KHI LAY MAU? ===")
print("(tai hien dung 2 buoc stratified split voi seed=42)")
print()
for name, spec in TASKS.items():
    chex = build_chexpert_cohort(spec, u_protocol="u_ignore", frontal_only=True)
    patients = (chex.groupby("patient_id")["label"].max().reset_index()
                    .rename(columns={"label":"patient_label"}))
    train_p, rest_p = _stratified_split(patients, 0.40, 42, "patient_label")
    val_p, test_p   = _stratified_split(rest_p, 0.50, 42, "patient_label")

    def pool(ids):
        s = chex[chex["patient_id"].isin(set(ids))].drop_duplicates("image_path")
        return int((s["label"]==1).sum()), int((s["label"]==0).sum())

    tp, tn = pool(train_p["patient_id"])
    vp, vn = pool(val_p["patient_id"])
    sp, sn = pool(test_p["patient_id"])
    print("%-18s TRAIN pool: pos=%-6d neg=%-6d | VAL pool: pos=%-5d neg=%-6d | TEST pool: pos=%-5d neg=%-6d"
          % (name, tp, tn, vp, vn, sp, sn))
