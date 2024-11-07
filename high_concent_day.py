import pandas as pd


a = pd.read_csv('python-code/lab/out_data/ホニキデータ/ラブリーちゃんfeetレンファ改.csv', header = 0, index_col = 0, encoding = 'shift-jis')
out = []
for i in range(2018,2021):
    for u in range(1,13):
        for s in range(1,32):
            string = str(i) + '-' + str(u).rjust(2, '0') + '-' + str(s).rjust(2, '0')
            b = a.filter(like = string, axis = 0)
            if len(b.values) != 0:
                max_index = b['江東区大島_OX'].idxmax()
                c = {'観測日': max_index, '日最高値': b['江東区大島_OX'].loc[max_index]}
                out.append(c)

out_df = pd.DataFrame(out)
out_df.to_csv('python-code/lab/out_data/ホニキデータ/江東区大島_OX_最高値.csv', index = False, encoding = 'shift-jis')