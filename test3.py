import pandas as pd
import glob

dir = "out_data/results/DNN/locationResult/"
file_name = "all_locateAndData.csv"

df = pd.read_csv(dir + file_name, index_col = 0, header = 0, encoding = "sjis")
df["調和平均"] = (2 * df["再現率"] * df["適合率"]) / (df["再現率"] + df["適合率"])
df["特徴量"] = [i.split("_")[0] for i in df.index.to_list()]
df.to_csv(dir + "test.csv", encoding = "sjis") 