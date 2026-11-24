import pandas as pd
df = pd.read_csv(r"CheXpert\train.csv")
cols = ["No Finding","Enlarged Cardiomediastinum","Cardiomegaly","Lung Opacity","Lung Lesion","Edema","Consolidation","Pneumonia","Atelectasis","Pneumothorax","Pleural Effusion","Pleural Other","Fracture","Support Devices"]
print("=== PHAN BO GIA TRI TUNG COT FINDING (train.csv, n=%d) ===" % len(df))
hdr = "%-28s%9s%9s%9s%9s" % ("column","1.0","0.0","-1.0","blank")
print(hdr)
for c in cols:
    v = df[c].value_counts(dropna=False)
    p1 = int(v.get(1.0,0)); p0 = int(v.get(0.0,0)); pm1 = int(v.get(-1.0,0))
    blank = int(df[c].isna().sum())
    print("%-28s%9d%9d%9d%9d" % (c,p1,p0,pm1,blank))
