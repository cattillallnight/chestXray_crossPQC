import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS
from qcalibnet.datasets import build_vindr_cohort, split_vindr_holdout, cohort_summary

print("=== VINDR: split_vindr_holdout tra ve NHUNG SPLIT NAO? ===")
c = build_vindr_cohort(TASKS["no_finding"], vote="any")
out = split_vindr_holdout(c, n_test=1000, seed=42, balanced=True)
print("cac key tra ve:", list(out.keys()))
for k, v in out.items():
    print("  %-14s n=%-6d split_label=%s" % (k, len(v), sorted(v["split"].unique())))
print()
print("=== TONG: 15000 anh duoc dung the nao? ===")
print("external_test :", len(out["external_test"]))
print("unused        :", len(out["unused"]))
print("TONG          :", len(out["external_test"]) + len(out["unused"]))
