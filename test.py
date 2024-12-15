from datetime import date
import pandas as pd
from datetime import timedelta

target = date(2019, 4, 1)
days = timedelta(days = 1)

df = pd.read_csv("out_data/results/DNN/上位20個/幸手_lag=1/out.csv", index_col = 0, header = 0)

for i in range(365):
    target_str = str(target)
    df_q = df[df.index.str.contains(target_str)]
    target += days
    print(df_q)





