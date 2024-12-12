import pandas as pd
from collections import Counter

df = pd.read_csv("out_data/results/DNN/results.csv", encoding = "sjis", index_col = 0, header = 0)
idxs = df.index.tolist()

# 元のリスト
original_list_1 = []
original_list_2 = []
original_list_3 = []

for idx in idxs:
    li = df.loc[idx, :].values.tolist()
    if li[-1] == 1:
        original_list_1 += li
    if li[-1] == 2:
        original_list_2 += li
    if li[-1] == 3:
        original_list_3 += li

# 要素の出現回数をカウント
counter_1 = Counter(original_list_1)
counter_2 = Counter(original_list_2)
counter_3 = Counter(original_list_3)

# 重複している要素のみを残す
duplicates_only_1 = [item for item, count in counter_1.items() if count > 7 and not pd.isna(item)]
duplicates_only_2 = [item for item, count in counter_2.items() if count > 8 and not pd.isna(item)]
duplicates_only_3 = [item for item, count in counter_3.items() if count > 6 and not pd.isna(item)]
all = duplicates_only_1 + duplicates_only_2 + duplicates_only_3
counter_all = Counter(all)
duplicates_only_all = [item for item, count in counter_all.items() if count > 2]

new_dict = {}

new_dict["lag=1"] = duplicates_only_1
new_dict["lag=2"] = duplicates_only_2
new_dict["lag=3"] = duplicates_only_3
new_dict["all"] = duplicates_only_all

le_li = []
for name in new_dict.keys():
    le = len(new_dict[name])
    le_li.append(le)
    
ma = max(le_li)

for name in new_dict.keys():
    while True:
        if len(new_dict[name]) < ma:
            new_dict[name].append(None)
        else:
            break

new_df = pd.DataFrame(new_dict).T
new_df.to_csv("out_data/results/DNN/results_dup.csv", encoding = "sjis")
