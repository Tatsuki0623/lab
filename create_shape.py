import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

dir_path = 'out_data/results/DNN/全データ'

dir_names = files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

for dir_name in dir_names:
    target_dir_path = 'out_data/results/DNN/全データ/' + dir_name + '/'
    target_shaps = glob(target_dir_path + 'shap_mean*.csv')
    
    for path in target_shaps:
        split_file_name = path.split('\\')[-1].split('_')
        if len(split_file_name) == 3:
            save_file_name = split_file_name[0] + '_10_' + split_file_name[1] + '_' + split_file_name[2]
            mean_target_path = target_dir_path + split_file_name[0] + '_' + split_file_name[2]
            df = pd.read_csv(mean_target_path, index_col = 0, header = 0)

        elif len(split_file_name) == 4:
            save_file_name = split_file_name[0] + '_10_' + split_file_name[1] + '_' + split_file_name[2] + '_' + split_file_name[3]
            mean_target_path = target_dir_path + split_file_name[0] + '_' + split_file_name[2] + '_' + split_file_name[3]
            df = pd.read_csv(mean_target_path, index_col = 0, header = 0)
        else:
            raise ValueError("ファイルの名前を見直しやがれ")
        
        pd.DataFrame(df.abs().mean()).sort_values(by = 0, ascending = False).head(10).to_csv(target_dir_path + save_file_name)