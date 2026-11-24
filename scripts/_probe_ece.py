import json, glob
from collections import defaultdict
import statistics as st
rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(r"results\tables\*_seed*.json"))]
by = defaultdict(list)
for r in rows: by[(r["task"], r["head"])].append(r)

print("=== ECE (raw) tren CheXpert test — mean +- std ===")
print("%-18s %-16s %-20s %-20s" % ("task","head","ECE raw","ECE calibrated"))
for task in ["no_finding","cardiomegaly","pleural_effusion","consolidation","pneumonia_proxy"]:
    for head in ["mlp","product_pqc","entangling_pqc"]:
        rs = by[(task,head)]
        e = [r["chexpert_test_raw_ece"] for r in rs]
        c = [r["chexpert_test_cal_ece"] for r in rs]
        print("%-18s %-16s %.4f +- %.4f     %.4f +- %.4f" % (task, head, st.mean(e), st.pstdev(e), st.mean(c), st.pstdev(c)))
    print()
print("=== epochs_run (co hoi tu khong?) ===")
for task in ["no_finding","cardiomegaly","pleural_effusion","consolidation","pneumonia_proxy"]:
    for head in ["mlp","product_pqc","entangling_pqc"]:
        rs = by[(task,head)]
        ep = [r["epochs_run"] for r in rs]
        print("%-18s %-16s epochs=%s" % (task, head, ep))
