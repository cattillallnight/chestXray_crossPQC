import json, glob, os
from collections import defaultdict
rows = []
for p in sorted(glob.glob(r"results\tables\*_seed*.json")):
    d = json.load(open(p, encoding="utf-8"))
    rows.append(d)
print("n runs:", len(rows))
print()
keys = sorted(rows[0].keys())
print("=== CAC KEY METRIC CO SAN ===")
for k in keys:
    print("  ", k)
