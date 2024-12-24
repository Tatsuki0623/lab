import pandas as pd
import glob

dir = "out_data/results/DNN/locationResult/"
file_name = "all_locateAndData.csv"
ss = "out_data/test_data/日別.csv"

df = pd.read_csv(dir + file_name, index_col = 0, header = 0, encoding = "sjis")
df_ss = pd.read_csv(ss, index_col = 0)
df