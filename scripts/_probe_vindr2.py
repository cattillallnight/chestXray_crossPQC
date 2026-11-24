import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS
from qcalibnet.datasets import build_vindr_cohort, cohort_summary

print("=== VINDR COHORT THEO TUNG TASK (sau khi gop annotation -> image-level) ===")
for name, spec in TASKS.items():
    c = build_vindr_cohort(spec, vote="any")
    s = cohort_summary(c)
    print("%-18s n=%-6d pos=%-6d neg=%-6d prevalence=%.4f" % (name, s["n"], s["n_pos"], s["n_neg"], s["prevalence"]))
