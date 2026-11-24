import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS, chexpert_task_label
from qcalibnet.datasets import load_chexpert_table

df = load_chexpert_table(Path(r"E:\cvchestpj\CheXpert\train.csv"))
print("raw CSV rows                    :", len(df))
print("exists=True                     :", int(df["exists"].sum()))
print("exists=False (anh thieu/hong)   :", int((~df["exists"]).sum()))
print()
d1 = df[df["Frontal/Lateral"] == "Frontal"]
print("sau frontal_only                :", len(d1), " (bo", len(df)-len(d1), "lateral)")
d2 = d1[d1["exists"]]
print("sau exists                      :", len(d2), " (bo", len(d1)-len(d2), ")")
print()
print("=== SO ANH BI DROP VI -1.0 (u_ignore), theo tung task ===")
for name, spec in TASKS.items():
    labs = [chexpert_task_label(r, spec, "u_ignore") for r in d2.to_dict("records")]
    n_none = sum(1 for l in labs if l is None)
    n_pos = sum(1 for l in labs if l == 1)
    n_neg = sum(1 for l in labs if l == 0)
    print("%-18s pos=%-7d neg=%-7d dropped(-1)=%-7d -> cohort=%d" % (name, n_pos, n_neg, n_none, n_pos+n_neg))
