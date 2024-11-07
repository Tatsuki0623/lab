import pandas as pd
import glob as gb

file_path_li = []
year_path = gb.glob('python-code/lab/input_data/coordinate/*.csv')
for path in year_path:
    file_path_li.append(path)

df_li = []

for path in file_path_li:
    coordinate_data = pd.read_csv(path, index_col = None, header = 0, encoding = 'cp932')
    out_col = ['国環研局番','測定局名','緯度','経度']

    coordinate_num = coordinate_data['国環研局番']
    coordinate_name = coordinate_data['測定局名']

    coordinate = []
    coordinate_vl_li = []
    for i in range(coordinate_data.shape[0]):

        coordinate_vl = []

        coordinate_v_num = coordinate_data.loc[i, '緯度_度'] + (coordinate_data.loc[i, '緯度_分'] / 60) + (coordinate_data.loc[i, '緯度_秒'] / 3600)
        coordinate_l_num = coordinate_data.loc[i, '経度_度'] + (coordinate_data.loc[i, '経度_分'] / 60) + (coordinate_data.loc[i, '経度_秒'] / 3600)

        coordinate_vl.append(coordinate_v_num)
        coordinate_vl.append(coordinate_l_num)

        coordinate_vl_li.append(coordinate_vl)



    coordinate_out_data_point = pd.DataFrame(coordinate_vl_li, index = None, columns = ['緯度','経度'], dtype = 'float64')
    coordinate_out_data_name = pd.concat([coordinate_num, coordinate_name], axis = 1)
    coordinate_out_data = pd.concat([coordinate_out_data_name, coordinate_out_data_point], axis = 1)

    df_li.append(coordinate_out_data)

out = df_li[0]

for point in range(1,len(df_li)):
    out = pd.concat([out, df_li[point]], axis = 0)
    
out.to_csv('python-code/lab/out_data/coordinate/csv/coordinate_kanto_2018.csv', index = False, columns = out_col)