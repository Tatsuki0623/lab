import pandas as pd
import folium

material_name = '対象地点'

df = pd.read_csv(f'out_data/coordinate/csv/{material_name}_coordinate.csv', index_col = None, header = 0)

#サイズを指定する
folium_figure = folium.Figure(width = 1500, height = 700)

# 初期表示の中心の座標を指定して地図を作成する。
center_lat = 140
center_lon = 36
folium_map = folium.Map([35.690921, 139.700258], zoom_start = 4.5).add_to(folium_figure)

# trainデータの300行目までマーカーを作成する。
for i in range(df.shape[0]):
    popup = str(df.loc[i, "測定局名"]) + '_' + str(df.loc[i, "国環研局番"])
    folium.Marker(location = [df.loc[i, "緯度"], df.loc[i, "経度"]], popup = popup).add_to(folium_map)
folium_map.save(f'out_data/coordinate/html/coordinate_{material_name}.html')