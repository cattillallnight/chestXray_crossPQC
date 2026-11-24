import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS
from qcalibnet.datasets import load_chexpert_table, build_chexpert_cohort, split_chexpert_patient_level

df = load_chexpert_table(Path(r"E:\cvchestpj\CheXpert\train.csv"))
print("=== POOL SAU KHI LOC FRONTAL + EXISTS (truoc khi sinh nhan) ===")
d = df[df["Frontal/Lateral"] == "Frontal"]
print("tong anh frontal:", len(d))
print()

print("=== KICH THUOC POOL TRAIN/VAL/TEST THEO PATIENT (seed 42) ===")
for name, spec in TASKS.items():
    chex = build_chexpert_cohort(spec, u_protocol="u_ignore", frontal_only=True)
    splits = split_chexpert_patient_level(chex, n_train=2000, n_val=500, n_test=500, seed=42, balanced_train=True)
    print("---", name, "---")
    for s in ("train","val","test"):
        f = splits[s]
        pos = int((f["label"]==1).sum()); neg = int((f["label"]==0).sum())
        print("   %-6s n=%-5d pos=%-6d neg=%-6d" % (s, len(f), pos, neg))
