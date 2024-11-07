import pandas as pd
'''
事前準備

1.　下記コマンドを実行すること
pip install pandas

2.　土地利用状況のcsvファイルを用意

3.　常時監視局の座標データのcsvファイルを用意

4.　2,3で用意したファイルのパスを取得（右クリックでパスをコピーを選択）

5.　実行

出力結果
debug_flagをTrueにすると、測定局のメッシュデータ(check.csv)と、3次メッシュ分のデータがない（格子が9個以下）のデータ(except.csv)を出力

基本は座標データに土地利用割合のからむが追加されたものが出力(result.csv)

予想エラー出現ヶ所 : 103行目、156行目
UnicodeDecodeErrorが出力された場合は encoding = 'sjis' の sjis を utf-8 か utf-16に変えると多分動く
FileNotFoundErorrが出力された場合はパスを見直して
'''

def extraction(station_dict):
    lon = station_dict['lon']
    lat = station_dict['lat']
    code = station_dict['測定局コード']

    target_grid = land_df.query(f"XMin <= {lon} <= XMax & YMin <= {lat} <= YMax")
    try:
        Xcenter, Ycenter = target_grid['col_number'].values[0], target_grid['row_number'].values[0]

        pulas_x = Xcenter + 1
        mainas_x = Xcenter - 1
        pulas_y = Ycenter + 1
        mainas_y = Ycenter - 1

        x00 = land_df.query(f'col_number == {mainas_x} & row_number == {pulas_y}')
        x01 = land_df.query(f'col_number == {Xcenter} & row_number == {pulas_y}')
        x02 = land_df.query(f'col_number == {pulas_x} & row_number == {pulas_y}')
        x10 = land_df.query(f'col_number == {mainas_x} & row_number == {Ycenter}')
        x12 = land_df.query(f'col_number == {pulas_x} & row_number == {Ycenter}')
        x20 = land_df.query(f'col_number == {mainas_x} & row_number == {mainas_y}')
        x21 = land_df.query(f'col_number == {Xcenter} & row_number == {mainas_y}')
        x22 = land_df.query(f'col_number == {pulas_x} & row_number == {mainas_y}')
        
        mesh = [target_grid, x00, x02, x01, x10, x12, x20, x21, x22]

        mesh_df = pd.concat(mesh)
        
        if debug_flag:
            if len(mesh_df) < 9:
                mesh[0].to_csv('except.csv', index = False, encoding = 'sjis')
                del mesh[0]
                mesh = [except_mesh.iloc[0].to_list() for except_mesh in mesh if not except_mesh.empty]

                for mesh_data in mesh:
                    with open('except.csv', 'a', encoding = 'sjis') as f:
                        f.write('\n' + (','.join(map(str, mesh_data))))

            cal_col = mesh_df.loc[:, ['田', 'その他の農用地', '森林', '荒地', '建物用地', '道路', '鉄道', 'その他の用地', '河川地及び湖沼', '海浜', '海水域', 'ゴルフ場']]

            corr_col = mesh_df.loc[:, ['XMin', 'XMax', 'YMin', 'YMax', 'XCenter', 'YCenter']]
            corr_df_li.append(corr_col)

        result = cal_rate(cal_col)
        result_df = pd.DataFrame(result, index = [0])
        return join_df(result_df, code)

    except Exception as e:
        print(f'({lon}, {lat})、{code}_{station_df[station_df['測定局コード'] == code]['測定局名'].values[0]}\n Show detail :\n {e}')

def join_df(pre_join_df, code):
    join_target = station_df[station_df['測定局コード'] == code]
    join_target.reset_index(drop = True, inplace = True)

    target_out_df = pd.concat([pre_join_df] * len(join_target), ignore_index = True)
    target_out_df = pd.concat([join_target, target_out_df], axis = 1)

    return target_out_df

def cal_rate(cal_col):
    sum_val = cal_col.sum()
    sum_cal_dict = dict(sum_val)
    sum_val = sum_val.sum()

    for key in sum_cal_dict.keys():
        rate = (sum_cal_dict[key] / sum_val) * 100
        
        sum_cal_dict[key] = rate
    
    return sum_cal_dict
    
def create_target_corrdinate_li(target_code):
    target_cal_corr = station_df.query(f'測定局コード == {target_code}').iloc[0, 5:8].to_dict()
    return target_cal_corr

def create_land_df(csv_path):
    target_data = pd.read_csv(csv_path, header = 0, encoding = 'sjis').astype('Float64')
    target_data_float = target_data['YCenter'].to_list()
    target_data['YCenter'] = target_data['YCenter'] * 10000
    target_data_int = target_data['YCenter'].astype(int).to_list()

    append_li = []
    for i in range(len(target_data_float)):
        if target_data_float[i] == target_data_int[i]:
            append_li.append(target_data_int[i]/10000)
        else:
            append_li.append(target_data_float[i])
    
    target_data['YCenter'] = append_li

    target_data = create_grid_num(target_data)

    return target_data

def create_grid_num(data):

    x_min_overall = data['XMin'].min()
    x_max_overall = data['XMax'].max()
    y_min_overall = data['YMin'].min()
    y_max_overall = data['YMax'].max()

    grid_width = data['XMax'][0] - data['XMin'][0]
    grid_height = data['YMax'][0] - data['YMin'][0]

    num_cols = int((x_max_overall - x_min_overall) / grid_width) + 1
    num_rows = int((y_max_overall - y_min_overall) / grid_height) + 1

    grid_indices = pd.DataFrame([
    (row, col) for row in range(num_rows) for col in range(num_cols)
    ], columns=['row_number', 'col_number'])

    data['col_number'] = ((data['XCenter'] - x_min_overall) / grid_width).astype(int)
    data['row_number'] = ((data['YCenter'] - y_min_overall) / grid_height).astype(int)

    return data

def out_csv(target_li):
    result_li = []

    result_li = list(map(extraction, target_li))
    out = pd.concat(result_li, ignore_index = True)
    out.to_csv('result.csv', index = False, encoding = 'sjis')

    if debug_flag:
        corr_df = pd.concat(corr_df_li, ignore_index = True)
        corr_df.to_csv('check.csv', index = False, encoding = 'sjis')

if __name__ == '__main__':
    # 測定局のcsvファイルのパスを入力
    station_df = pd.read_csv('測定局のcsvファイルのパス', header = 0, encoding = 'sjis')

    # 土地利用状況のcsvファイルのパスを入力
    land_df = create_land_df(csv_path = '土地利用状況のcsvファイルのパス')

    target_cal_code = list(set(station_df['測定局コード'].to_list()))
    target_li = list(map(create_target_corrdinate_li, target_cal_code))

    debug_flag = True
    
    if debug_flag:
        corr_df_li = []
        except_li = []

    out_csv(target_li = target_li)


