import pandas as pd

# CSVファイルの読み込み
df = pd.read_csv("C:/Users/yuyun/Downloads/世田谷区世田谷.csv", encoding='utf-8')

df = df.T
df.to_csv("C:/Users/yuyun/Downloads/new世田谷区世田谷.csv", encoding='utf-8')

# code_list = df.drop(["年","観測月","観測日"], axis = 1).columns.to_list()

# # 4月〜8月の範囲で日付リストを作成
# date_range = pd.date_range(start='2018-04-01', end='2020-08-31')
# filtered_dates = date_range[(date_range.month >= 4) & (date_range.month <= 8)]
# date_time_list = filtered_dates.strftime('%Y/%m/%d').tolist()

# # 各測定局コードと日付に対して最大値を計算する関数

# out_list = []
# for u in date_time_list:
#     target = df.loc[df.index == u]
#     max_df = target.loc[:, code_list].max()
#     out_list.append(max_df)

# out_df = pd.DataFrame(out_list, index = date_time_list).reset_index(names = '観測日時')
# out_df.to_csv('C:/Users/yuyun/Desktop/関西NMHC_日最高値.csv', index = False)