import pandas as pd
import glob

path_li = glob.glob('out_data/test_data/モデル評価*')
for path in path_li:
    df = pd.read_csv(path, index_col = 0, header = 0)
    df["特徴量"] = [i.split("_")[0] for i in df.index.to_list()]
    df["地点"] = [u.split("_")[1] for u in df.index.to_list()]
    df["lag"] = [n.split("=")[-1] for n in df.index.to_list()]
    df.to_csv(path)