import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS, VINDR_PNEUMONIA_PROXY
from qcalibnet.datasets import load_vindr_annotations, _is_real_image
from qcalibnet.paths import vindr_image_path

ann = load_vindr_annotations()
print("=== BUOC 1: gop annotation -> set class_name theo image_id ===")
grouped = ann.groupby("image_id")["class_name"].apply(lambda s: set(s))
print("so image_id:", len(grouped))
print()
print("=== BUOC 2: kiem tra anh that (_is_real_image) ===")
missing = 0
for iid in grouped.index:
    if not _is_real_image(vindr_image_path(iid)):
        missing += 1
print("anh bi loai (thieu/hong/._):", missing)
print("anh con lai:", len(grouped) - missing)
print()
print("=== BUOC 3: kiem tra 'No finding' co bi lan voi class khac khong ===")
both = 0
for iid, cls in grouped.items():
    if "No finding" in cls and len(cls) > 1:
        both += 1
print("anh vua co 'No finding' vua co class benh khac:", both)
print()
print("=== BUOC 4: so class moi anh ===")
ncls = grouped.apply(len)
print(ncls.value_counts().sort_index().to_dict())
