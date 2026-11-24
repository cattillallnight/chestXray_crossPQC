import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS
from qcalibnet.datasets import build_vindr_cohort, split_vindr_holdout, cohort_summary

print("=== VINDR: LAY MAU 1000 ANH (balanced vs natural), seed 42 ===")
for name, spec in TASKS.items():
    c = build_vindr_cohort(spec, vote="any")
    bal = split_vindr_holdout(c, n_test=1000, seed=42, balanced=True)["external_test"]
    nat = split_vindr_holdout(c, n_test=1000, seed=43, balanced=False)["external_test"]
    sb, sn = cohort_summary(bal), cohort_summary(nat)
    print("%-18s BAL pos=%-5d neg=%-5d prev=%.3f | NAT pos=%-5d neg=%-5d prev=%.3f"
          % (name, sb["n_pos"], sb["n_neg"], sb["prevalence"], sn["n_pos"], sn["n_neg"], sn["prevalence"]))
