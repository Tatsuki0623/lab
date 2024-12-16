import json
import pandas as pd
import netCDF4 as nc
import glob
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# NCファイルの読み込み関数
def nc_to_json(nc_file_path):
    # NetCDFファイルを開く
    # 全角のパスに対応していない
    dataset = nc.Dataset(nc_file_path, 'r')
    
    # NetCDFのデータ構造をJSON互換の辞書に変換
    def convert_to_dict(dataset):
        result = {}
        for key in dataset.variables.keys():
            if key != 'TFLAG':
                # 各変数のデータとその属性を取得
                variable_data = dataset.variables[key][:]
                attributes = {attr: dataset.variables[key].getncattr(attr) for attr in dataset.variables[key].ncattrs()}
                result[key] = {
                    "data": variable_data.tolist(),  # 配列データをリストに変換
                    "attributes": attributes
                }
        return result
    
    data_dict = convert_to_dict(dataset)
    # NetCDFファイルを閉じる
    dataset.close()

    return data_dict

def test(data_dict, nc_file, flag):
    # グリッドデータを格納するリスト
    if flag:
        
        keys = data_dict.keys()
        print(keys)
        print('出力された物質からターミナルにコピペしてエンターを押してください')

        global select_keys
        select_keys = input().replace("'", '').replace(' ', '').split(',')


    for material_name in select_keys:
        # 各時間について処理
        grid_data_list = []
        for u in range(24):
            # 縦軸と横軸のデータを取得
            vertical_data = data_dict[material_name]['data'][u][0]
            horizontal_data = data_dict[material_name]['data'][u][0][0]
            
            # 縦48行、横69列に変換
            grid_data = np.array(vertical_data)[:len(vertical_data), :len(horizontal_data)]

            grid_data_list.append(grid_data)

        if "d04" in nc_file.split('\\')[-1]:
            min_lat, max_lat = 34.308, 36.878  # 緯度の範囲
            min_lon, max_lon = 138.502, 141.381  # 経度の範囲
            num_rows, num_cols = 57, 51
        elif "d03" in nc_file.split('\\')[-1]:
            min_lat, max_lat = 33.581, 35.892  # 緯度の範囲
            min_lon, max_lon = 133.798, 137.520  # 経度の範囲
            num_rows, num_cols = 47, 69
        else:
            print("失敗")
            continue

        process_material(grid_data_list, material_name, min_lat, max_lat, min_lon, max_lon, num_rows, num_cols)

# ファイルごとにGeoJSON生成を並列処理で実行
def process_material(grid_data_list, material_name, min_lat, max_lat, min_lon, max_lon, num_rows, num_cols):
    latitudes = np.linspace(min_lat, max_lat, num_rows)  # 下から上へ
    longitudes = np.linspace(min_lon, max_lon, num_cols)  # 左から右へ

    year = save_name[2:6]
    month = save_name[6:8]
    day = save_name[8:]

    features = []

    # 各時間のグリッドデータについて処理
    for hour, grid in enumerate(grid_data_list):
        for i in range(num_rows):  # 縦方向
            for j in range(num_cols):  # 横方向
                value = grid[i, j]
                substance = material_name  # 例としてNO2を設定
                year = save_name[2:5]
                month = save_name[6:7]
                day = save_name[8:]      # 必要に応じて設定
                
                # 各ポイントのFeatureを作成
                point = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(longitudes[j]), float(latitudes[i])]
                    },
                    "properties": {
                        "value": float(value),
                        "substance": substance,
                        "year": year,
                        "month": month,
                        "day": day,
                        "hour": hour
                    }
                }
                features.append(point)

    # GeoJSONのFeatureCollectionを作成
    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    # GeoJSONファイルとして書き出し
    with open(dir_path + f'/geojson/grid_data_{save_name}_{material_name}.geojson', 'w') as f:
        json.dump(geojson_data, f, indent=2)


user_name = 'yuyun'
print('日付をyyyymmdd(例：20180301)をターミナルに入力後エンターを押してください')
select_date = input()
flag = True
select_keys = []

# ディレクトリパス
dir_path = f'C:/Users/{user_name}/Desktop/dir'

file_li = glob.glob(f'{dir_path}/*_{str(select_date)}*.nc')

for i in file_li:
    nc_file = i
    save_name = re.sub(r'\D', '', i)
    data_dict = nc_to_json(nc_file)
    test(data_dict, nc_file, flag)
    flag = False