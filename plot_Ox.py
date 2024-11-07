import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#8時間移動平均値を出力
def mean_8(material_df: pd.DataFrame):

    index_num = material_df.shape[0] + 1
    material_df_8_mean_li = []

    for i in range(index_num):
        start_point = i
        last_point = i + 7

        target_mean_data = material_df[start_point : last_point].to_numpy()
        target_mean_data_elements = target_mean_data.shape[0]

        if target_mean_data_elements == 7:
            mean = np.mean(target_mean_data)
            material_df_8_mean_li.append(mean)

        else:
            material_df = material_df.drop(index = material_df.index[start_point:])
            for u, material_df_8_mean in enumerate(material_df_8_mean_li):
                material_df.iloc[u] = material_df_8_mean

            break
    
    return material_df

a = pd.read_csv('python-code/lab/input_data/learning_data/machine_learning_harumi.csv', header = 0, index_col = 0)
a = a['OX']

s = mean_8(a)

plt.figure(figsize = (14, 7))
plt.plot(a.index[0:s.shape[0]], s)
plt.show()
