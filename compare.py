import pandas as pd
from glob import glob

df = pd.read_csv("out_data/results/DNN/com_features.csv", encoding = "sjis", index_col = 0, header = 0)
cols = df.columns.tolist()
idxs = df.index.tolist()
new_df_dict = {}


for col in cols:
    for idx in idxs:
        val = df.loc[idx, col]
        re_val = val.replace('[', '').replace(']', '').replace("'","").replace(' ', '').replace(':', '_').split(',')
        new_df_dict[idx + "_" + col] = re_val

le_li = []
for name in new_df_dict.keys():
    le = len(new_df_dict[name])
    le_li.append(le)
    
ma = max(le_li)

for name in new_df_dict.keys():
    while True:
        if len(new_df_dict[name]) < ma:
            new_df_dict[name].append(None)
        else:
            break

locate = list(set([i.split("_")[0] for i in cols]))
new_df = pd.DataFrame(new_df_dict).T
new_df["地点"] = [i.split("_")[0] for i in new_df.index.to_list()]
new_df["lag"] = [i.split("_")[1].split("=")[-1] for i in new_df.index.to_list()]
new_df.to_csv("out_data/results/DNN/com.csv", encoding = "sjis")

