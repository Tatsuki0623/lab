import pandas as pd
from datetime import date
from datetime import timedelta

df = pd.read_csv("out_data/results/DNN/god/世田谷_lag=1.csv", index_col = 0, header = 0)
target_day = date(2019, 5, 24)
target_str = str(target_day)
target_df = df[df.index.str.contains(target_str)]
print(target_df)