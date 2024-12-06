import pandas as pd
import os

dir_path = "out_data/results/DNN/"
top_dir_names = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
df_list = []

for target_sub_dir in top_dir_names:
    target_dir_path = dir_path + target_sub_dir + "/"
    sub_dir_names = [f for f in os.listdir(target_dir_path) if os.path.isdir(os.path.join(target_dir_path, f))]
    
    for target_dir in sub_dir_names:
        
        try:
            target_file_path = target_dir_path + target_dir + "/high_cooncent_check.csv"
            df = pd.read_csv(target_file_path, index_col = 0).T
            df.rename(index = {"0": target_sub_dir + "_" + target_dir}, inplace = True)
        except FileNotFoundError:
            try:
                target_file_path = target_dir_path + target_dir + "/high_concent_check.csv"
                df = pd.read_csv(target_file_path, index_col = 0).T
                df.rename(index = {"0": target_sub_dir + "_" + target_dir}, inplace = True)
            except FileNotFoundError:
                df = None
        print(df)
        if df is not None:
            continue
        df_list.append(df)

new_df = pd.concat(df_list)
new_df.drop(columns = ['予測高濃度出現回数', '予測高濃度追跡', '予測高濃度追跡率', '適合率'], inplace = True)

locates = set([i.split("_")[0] for i in target_sub_dir])
for locate in locates:
    filtered_df = new_df[new_df.index.str.contains(locate)]
    filtered_df.sort_index(inplace=True)
    print(filtered_df)
