import json, glob
from collections import defaultdict
import statistics as st

rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(r"results\tables\*_seed*.json"))]
by = defaultdict(list)
for r in rows:
    by[(r["task"], r["head"])].append(r)

print("=== CHEXPERT TEST AUROC (raw) — mean +- std qua 3 seed ===")
print("%-18s %-16s %-22s %-10s" % ("task","head","AUROC mean+-std","n_trainable"))
for task in ["no_finding","cardiomegaly","pleural_effusion","consolidation","pneumonia_proxy"]:
    for head in ["mlp","product_pqc","entangling_pqc"]:
        rs = by[(task,head)]
        a = [r["chexpert_test_raw_auroc"] for r in rs]
        nt = rs[0]["n_trainable"]
        print("%-18s %-16s %.4f +- %.4f      %d" % (task, head, st.mean(a), st.pstdev(a), nt))
    print()
