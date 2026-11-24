import sys, pandas as pd
from pathlib import Path
sys.path.insert(0, r"E:\cvchestpj")
from qcalibnet.labels import TASKS, VINDR_CLASS_ID
from qcalibnet.datasets import load_vindr_annotations, build_vindr_cohort, _is_real_image
from qcalibnet.paths import vindr_image_path

ann = load_vindr_annotations()
print("=== VINDR ANNOTATION THO ===")
print("tong dong annotation:", len(ann))
print("so image_id duy nhat:", ann["image_id"].nunique())
print("so rad_id (bac si):", ann["rad_id"].nunique(), sorted(ann["rad_id"].unique()))
print()
print("=== SO ANNOTATION MOI ANH (phan bo) ===")
per = ann.groupby("image_id").size()
print(per.describe())
print()
print("=== ANH CO NHIEU BAC SI CUNG GAN (multi-radiologist) ===")
rad_per_img = ann.groupby("image_id")["rad_id"].nunique()
print(rad_per_img.value_counts().sort_index().to_dict())
