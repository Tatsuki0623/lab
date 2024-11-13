import pandas as pd

# CSVファイルの読み込み
low_df = pd.read_csv("out_data/results/DNN/AllFeatures_low.csv", encoding='utf-8', index_col=0)
high_df = pd.read_csv("out_data/results/DNN/AllFeatures.csv", encoding='utf-8', index_col=0)

low_dict = low_df.to_dict()['0']
high_dict = high_df.to_dict()['0']

new_dict = {}

for name, features_name in low_dict.items():
    low_features = features_name.replace('[', '').replace(']', '').replace("'","").replace(' ', '').split(',')
    high_features = high_dict[name].replace('[', '').replace(']', '').replace("'","").replace(' ', '').split(',')

    new_li = list(set(low_features + high_features))
    print(len(new_li))
    new_dict[name] = str(new_li)
    print(new_li)

a = pd.DataFrame(new_dict, [0]).T
a.to_csv('out_data/results/DNN/mergefeatures.csv')