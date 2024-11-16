import pandas as pd
import glob as gb

csv_join = gb.glob('out_data/ホニキデータ/*.csv')
path = []

print(csv_join)

name = []

for n in csv_join:
    n = n.split('\\')[-1].split('.')[0]
    name.append(n)


for i, u in enumerate(csv_join):
    if i == 0:
        target_df = pd.read_csv(u, header = 0, index_col = 0, encoding = 'shift-jis')
        col = target_df.columns
        
        target_df = target_df.rename(columns = {col[0]: name[i]})

    else:
        join_df = pd.read_csv(u, header = 0, index_col = 0, encoding = 'shift-jis')
        col = join_df.columns
        
        join_df = join_df.rename(columns = {col[0]: name[i]})

        target_df = pd.concat([target_df, join_df], axis = 1)

target_df.sort_index(inplace = True)
target_df.to_csv('out_data/ホニキデータ/merge_NOx_value.csv', encoding = 'shift-jis')