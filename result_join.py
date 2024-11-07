from random import randint as rd
import pandas as pd
from glob import glob

df_li = []
file_num = [i for i in range(1,25)]

features = []
li1 = 'SPM：TEMP：WD：PM25：sin_day：cos_day：sin_hour：cos_hour'.split('：')
li2 = 'PM25：SO2：SPM：TEMP：WD：cos_day：cos_hour：sin_day：sin_hour'.split('：')

features.append(li1)
features.append(li2)
for i in file_num:

    if i < 16:
        u = features[0]
    else:
        u = features[1]
    l = ('：'.join(u))
    out_df = pd.read_csv(f'python-code/lab/out_data/results/out_csv/harumi_timesteps={i}_scalermode=0_list={l}_出力結果.csv', header = 0)
    df_li.append(out_df)
out = pd.concat(df_li)
out.index = file_num
out.to_csv('python-code/lab/out_data/results/harumi/out_csv/harumi_timestep=join_scalermode=0_list=SPM：TEMP：WD：PM25：sin_day：cos_day：sin_hour：cos_hour_出力結果.csv', encoding = 'shift-jis')
print(out)