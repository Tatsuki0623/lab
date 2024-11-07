import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob

dir_path = 'python-code/lab/out_data/test_data'

dir_names = files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

for dir_name in dir_names:
    target_dir_path = 'python-code/lab/out_data/test_data/' + dir_name + '/'
    target_shaps = glob(target_dir_path + 'shap_mean*.csv')
    
    for path in target_shaps:
        split_file_name = path.split('\\')[-1].split('_')
        if len(split_file_name) == 3:
            save_file_name = split_file_name[0] + '_20_' + split_file_name[1] + '_' + split_file_name[2]
            mean_target_path = target_dir_path + split_file_name[0] + '_' + split_file_name[2]
            df = pd.read_csv(mean_target_path, index_col = 0, header = 0)

        elif len(split_file_name) == 4:
            save_file_name = split_file_name[0] + '_20_' + split_file_name[1] + '_' + split_file_name[2] + '_' + split_file_name[3]
            mean_target_path = target_dir_path + split_file_name[0] + '_' + split_file_name[2] + '_' + split_file_name[3]
            df = pd.read_csv(mean_target_path, index_col = 0, header = 0)
        else:
            raise ValueError("ファイルの名前を見直しやがれ")
        
        #pd.DataFrame(df.abs().mean()).sort_values(by = 0, ascending = False).head(20).to_csv(target_dir_path + save_file_name)
        corr_matrix = df.corr()

        # ヒートマップの描画
        plt.figure(figsize=(12, 10))  # 図のサイズを調整
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt=".2f", 
                    cbar=True, square=True, vmin=-1, vmax=1)
        plt.title("Feature Correlation Heatmap")
        plt.show()

"""         # 表示するセルのサイズを設定
        chunk_size = 24  # 分割サイズ (例: 20×20)

        # 行と列を範囲でループし、部分行列を描画
        for i in range(0, corr_matrix.shape[0], chunk_size):
            for j in range(0, corr_matrix.shape[1], chunk_size):
                plt.figure(figsize=(8, 6))
                sns.heatmap(corr_matrix.iloc[i:i+chunk_size, j:j+chunk_size],
                            cmap='coolwarm', annot=False, fmt=".2f", 
                            cbar=True, square=True, vmin=0, vmax=1,
                            xticklabels=corr_matrix.columns[j:j+chunk_size],
                            yticklabels=corr_matrix.index[i:i+chunk_size])
                plt.title(f"Feature Correlation Heatmap (Rows {i} to {i+chunk_size}, Columns {j} to {j+chunk_size})")
                plt.show() """

