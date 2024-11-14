import pandas as pd

# CSVファイルの読み込み
low_df = pd.read_csv("out_data/results/DNN/AllFeatures_low.csv", encoding='utf-8', index_col=0)
high_df = pd.read_csv("out_data/results/DNN/AllFeatures.csv", encoding='utf-8', index_col=0)

low_dict = low_df.to_dict()['0']
high_dict = high_df.to_dict()['0']

new_dict = {}
high_pre_dict = {}

for name, features_name in low_dict.items():
    low_features = features_name.replace('[', '').replace(']', '').replace("'","").replace(' ', '').split(',')
    high_features = high_dict[name].replace('[', '').replace(']', '').replace("'","").replace(' ', '').split(',')

    new_li = list(set(low_features + high_features))
    new_dict[name] = new_li
    high_pre_dict[name] = high_features

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

print(new_dict)

a = pd.DataFrame(new_dict)
print(a)
a.to_csv('out_data/results/DNN/mergefeatures_df.csv')

b = pd.DataFrame(high_pre_dict)
print(b)
b.to_csv('out_data/results/DNN/Allfeatures_high_df.csv')