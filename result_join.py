import pandas as pd
import os

dir_path = "out_data/results/DNN/"
top_dir_names = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
df_list = []

for target_sub_dir in top_dir_names:
    target_dir_path = dir_path + target_sub_dir + "/"
    sub_dir_names = [f for f in os.listdir(target_dir_path) if os.path.isdir(os.path.join(target_dir_path, f))]
    
    for target_dir in sub_dir_names:
        target_file_path = target_dir_path + target_dir + "/high_concent_check.csv"
        df = pd.read_csv(target_file_path, index_col = 0)
        df.rename(index = {"0": target_sub_dir}, inplace = True)
        df["lag"] = target_dir.split("=")[-1]
        df["locate"] = target_dir.split("_")[0]
        df_list.append(df)

new_df = pd.concat(df_list)
new_df.to_csv("out_data/results/DNN/location_result/all_locate.csv", encoding="sjis")

locates = set([i.split("_")[0] for i in sub_dir_names if (len(i.split("_")) == 2)])
for locate in locates:
    filtered_df = new_df[locate == new_df["locate"]]
    filtered_df.sort_index(inplace=True)
    filtered_df.to_csv(f"out_data/results/DNN/location_result/{locate}.csv", encoding="sjis")
