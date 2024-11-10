import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from glob import glob
from sklearn.metrics import root_mean_squared_error

dir_path = 'out_data/results/DNN'

dir_names = files_dir = [
    f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))
]

for dir_name in dir_names:
    target_dir_path = 'out_data/results/DNN/' + dir_name + '/'
    target_shaps = glob(target_dir_path + 'out.csv')
    
    for path in target_shaps:
        target_df = pd.read_csv(path, index_col = 0)
        target_predict = target_df.query('predict >= 80')
        predict_len = len(target_predict)
        obs_len = len(target_predict.query('obs >= 80'))
        predict_per = (obs_len / predict_len) * 100


        rmse = root_mean_squared_error(target_df['obs'], target_df['predict'])
        high_out = target_df.query('obs >= 80')
        high_concent_day = high_out.index.to_list()
        high_out_len = len(high_out)
        high_rmse = root_mean_squared_error(high_out['obs'], high_out['predict'])
        high_out_p = len(high_out.query('predict >= 80'))
        obs_per = (high_out_p / high_out_len) * 100

        compa = (2 * obs_per * predict_per) / (obs_per + predict_per)

        high_concent_dict = {
                            '高濃度出現回数': high_out_len,
                            '高濃度追跡': high_out_p,
                            '高濃度追跡率': obs_per,
                            '高濃度RMSE': high_rmse,
                            '予測高濃度出現回数': predict_len,
                            '予測高濃度追跡': obs_len,
                            '予測高濃度追跡率': predict_per,
                            '適合率': compa
                           }
        
        out = pd.DataFrame(high_concent_dict, [0])
        out.to_csv(target_dir_path + 'new_high_concent_check.csv')
        
        
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

