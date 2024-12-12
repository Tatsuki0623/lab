import pandas as pd 

df = pd.read_csv("out_data/test_data/large_dataset.csv", encoding = "sjis", index_col = 0, header = 0)
df_q = df.query("10000000 <= annual_income")
print(df_q.shape)