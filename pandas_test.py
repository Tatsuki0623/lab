import pandas as pd


a = pd.read_csv('python-code/lab/out_data/ホニキデータ/ラブリーちゃん改.csv', header = 0, index_col = 0)
out = []
for i in range(2018,2021):
    for u in range(1,13):
        for s in range(1,32):
            string = str(i) + '-' + str(u).rjust(2, '0') + '-' + str(s).rjust(2, '0')
            b = a.filter(like = string, axis = 0)
            if len(b.values) != 0:
                max_index = b['13108010_江東区大島_NMHC'].idxmax()
                c = {'観測日時': max_index, '日最高値': b['13108010_江東区大島_NMHC'].loc[max_index]}
                out.append(c)

out_df = pd.DataFrame(out)
out_df.set_index('観測日時', inplace = True)
out_df['日最高値1'] = out_df.shift(-1)
out_df['差分'] = out_df['日最高値'] - out_df['日最高値1']
print(out_df)