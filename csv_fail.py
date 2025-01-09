import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob as gb
import datetime

#csvファイルの読み込み
def csv_read(header_label_point = 0, index_label_point = 1, start_year = 2009, last_year = 2021, locate = 'tokyo'):
    file_path_li = []
    last_year += 1

    for year in range(start_year, last_year):
        #inpu_dataフォルダ直下にあるフォルダに入っているcsvをすべて読み込み
        #フォルダ直下のフォルダのパス名は西暦（YYYY）にすること
        year_path = gb.glob(f'input_data/kankyou_date/{locate}/' + str(year) + '/*.csv')
        for path in year_path:
            file_path_li.append(path)

    df_li = {}

    for i in file_path_li:
        index_point = 0
        measurement_year_point = 0
        target_items_point = 2

        #読み込んだcsvファイルをすべてpandasデータに変換
        df = pd.read_csv(i, header = header_label_point, index_col = index_label_point, encoding = 'cp932')

        year = df.iat[index_point, measurement_year_point]
        target = df.iat[index_point, target_items_point].strip()
        
        #読み込んだpandasデータをdict形式でdf_liに格納
        #key名は西暦(YYYY)_物質名   例：2018_OX        
        item_name = str(year) + '_' + str(target)

        df_li[item_name] = df
    
    return df_li

#欠損値の線形保管
def missing_value_imputation(df_li, interpolation_flag = True):

    for df_li_key, target_df in df_li.items():

        # "9998" を NaN に置き換え
        target_df.replace(9998, np.nan, inplace = True)

        # 未測定日（全列が NaN）を除外
        target_df = target_df.dropna(how='all', subset = target_df.columns[7:])

        # "9999"、"9997" を NaN に置き換え
        target_df = target_df.replace(9999, np.nan)
        target_df = target_df.replace(9997, np.nan)
        
        if df_li_key in 'SUN':
            #日射量に関して欠損地を0に変換する
            target_df = target_df.replace(np.nan, 0)

        else:
            if interpolation_flag:
                # 数値部分を取り出してフラットにする
                numeric_columns = target_df.columns[6:]
                numeric_data = target_df[numeric_columns].values.flatten()

                # 欠損値を線形補完
                numeric_data_interpolated = pd.Series(numeric_data).interpolate(method = 'linear', limit_direction = 'both')

                # 補完したデータを元の形に戻す
                reshaped_data = numeric_data_interpolated.values.reshape(target_df[numeric_columns].shape)
                target_df.loc[:,numeric_columns] = reshaped_data
            
            else:
                #欠損地を0に変換
                target_df = target_df.replace(np.nan, 0)
            

        df_li[df_li_key] = target_df
    
    return df_li

#データのフィルタ
def extraction(df_li, query_items = list):

    for df_li_key, target_df in df_li.items():
        query_df = target_df

        for i,query_item in enumerate(query_items):
            if i != 0:
                operator = query_item[0]

                if operator == 0:
                    query_df = query_df.query(f'{query_item[1]} == {query_item[2]}')

                elif operator == 1:
                    query_df = query_df.query(f'{query_item[1]} in {query_item[2]}')
                
                elif operator == 2:
                    query_df = query_df.query(f'{query_item[1]} <= {query_item[2]}')
                
                elif operator == 3:
                    query_df = query_df.query(f'{query_item[1]} >= {query_item[2]}')
                
                elif operator == 4:
                    query_df = query_df.query(f'{query_item[1]} >= {query_item[2]} and {query_item[1]} <= {query_item[3]}')
            
        df_li[df_li_key] = query_df
    return df_li

#時系列データの作成
def year_correction(df_li):
    
    #2018年4月~2019年3月までのデータが2018年4月~2018年3月という形なので年が変わったら+1年する処理
    for df_li_key, target_df in df_li.items():
        target_df.loc[target_df['測定月'].isin([1, 2, 3]), '測定年度'] += 1
    
        df_li[df_li_key] = target_df
    
    return df_li

#一時間ごとのデータセットラベルの作成
def create_label_name(df_li, target_columns_point_li = list):
    connecting_character = '/'
    new_index_name_dict = {}
    for df_li_key, target_format_df in df_li.items():
        columns_name_li = []
        new_index_name_li = []

        #対象物質名のリスト作成
        for target_columns_point in target_columns_point_li:
            columns_name = target_format_df.columns[target_columns_point]
            
            columns_name_li.append(columns_name)
        
        columns_name_li_elements = len(columns_name_li)
        columns_name_li_last_elements = len(columns_name_li) - 1
        
        #日付データが年、月、日になっているためそれらをすべて結合する（YYYY/MM/DDを作成）
        for target_columns_num in range(columns_name_li_elements):

            if target_columns_num == 0:
                cat_df = target_format_df[columns_name_li[target_columns_num]].astype('str') + connecting_character
            
            elif target_columns_num == columns_name_li_last_elements:
                cat_df += target_format_df[columns_name_li[target_columns_num]].astype('str')
            
            else:
                connect_cat_df = target_format_df[columns_name_li[target_columns_num]].astype('str') + connecting_character               
                cat_df += connect_cat_df

        hour = [str(u) + '時' for u in range(24)]

        #上で作った日付データ一日ごとに0~23時の時間値を付与
        for date in cat_df:
            for time in hour:
                new_index_name = date + '_' + time
                new_index_name_li.append(new_index_name)
        
        new_index_name_dict[df_li_key] = new_index_name_li
            
    return new_index_name_dict

#上記のデータラベルと対象物質データから新しいデータフレームを作成
def new_index_date_df(df_li, new_index_name_dict, flag = False):

    for df_li_key, target_df in df_li.items():
        target_df_concent_value = target_df[target_df.columns[6:]].values
        target_df_concent_value_size = target_df_concent_value.size
        target_df_concent_value_np = np.reshape(target_df_concent_value, (target_df_concent_value_size, 1))

        target_material = df_li_key.split('_')[-1]

        new_columns_name_li = [target_material]
        
        new_df = pd.DataFrame(target_df_concent_value_np, index = new_index_name_dict[df_li_key], columns = new_columns_name_li)

        df_li[df_li_key] = new_df
    
    #flagを立たせるとこの時点でできたデータを出力させる
    if flag:
        for i in df_li.keys():
            df_li[i].to_csv(f'out_data/new_index_{i}.csv', header = False)

    return df_li


def join(df_li = dict):
    
    df_li_keys = df_li.keys()
    df_year_data = [int(year.split('_')[0]) for year in df_li_keys]

    year_df_li = {}


    for i, df_li_key in enumerate(df_li_keys):
        if i == 0:
            join_df = df_li[df_li_key]
            join_df.index = pd.to_datetime(join_df.index, format='%Y/%m/%d_%H時')
        
        else:
            new_df = df_li[df_li_key]
            new_df.index = pd.to_datetime(new_df.index, format='%Y/%m/%d_%H時')

            year = df_year_data[i]
            pre_year = df_year_data[i - 1]
            
            #年ごとにデータフレームを作成する
            if year == pre_year:
                col = new_df.columns[0]
                
                #新しい列として追加
                join_df[col] = new_df[col]

            else:
                #作ったデータフレームをdict形式で保存
                year_df_li[df_year_data[i - 1]] = join_df

                #join_dfのインスタンス化
                join_df = df_li[df_li_key]
                join_df.index = pd.to_datetime(join_df.index, format='%Y/%m/%d_%H時')
    
    #dictへ格納したデータフレームを下に結合させていく
    if len(year_df_li) >= 1:
        out_df = year_df_li[df_year_data[0]]
        year_df_li[df_year_data[i]] = join_df
    
        for i, year_df_li_key in enumerate(year_df_li.keys()):
            if i != 0:
                out_df = pd.concat([out_df, year_df_li[year_df_li_key]], axis = 0)
    else:
        #もし一年分のみであればそのまま出力
        out_df = join_df

    #日付を降順でソート
    out_df = out_df.sort_index()

    out_df['year'] = out_df.index.year
    out_df['month'] = out_df.index.month
    out_df['day'] = out_df.index.day
    out_df['time'] = out_df.index.hour

    return out_df

#出力するデータフレームを日時でフィルターにかける
def df_filter(target_df, start_date, end_date):
    start_date_li = list(map(int, start_date.split('/')))
    end_date_li = list(map(int, end_date.split('/')))

    start_year = start_date_li[0]
    start_month = start_date_li[1]
    start_day = start_date_li[2]
    start_hour = start_date_li[3]

    end_year = end_date_li[0]
    end_month = end_date_li[1]
    end_day = end_date_li[2]
    end_hour = end_date_li[3]

    target_df = target_df.query(f'{start_year} <= year <= {end_year} & {start_month} <= month <= {end_month} & {start_day} <= day <= {end_day} & {start_hour} <= time <= {end_hour}')
    target_df.dropna(how = 'all', axis = 1, inplace = True)
    

    return target_df


def df_extraction(out_df, target_times):
    df_li = []
    for time in target_times:
        query_df = out_df.query(f"time == {time}")
        df_li.append(query_df)
    
    new_df = pd.concat(df_li, axis=0)
    new_df.sort_index(inplace=True)

    return new_df

#時間を横積みに変換する
def df_T(target_df: pd.DataFrame):
    col_name = target_df.columns.to_list()

    target_df_li = []
    for name in col_name:
        target_df['date'] = target_df.index.date
        # ピボットテーブルを使って一日ごとに横展開
        df_pivot = target_df.pivot(index = 'date', columns = 'time', values = name)

        # 列名を変更（CH4_0, CH4_1, ... CH4_23）
        df_pivot.columns = [f'{name}_{i}' for i in df_pivot.columns]
        target_df_li.append(df_pivot)
        
#対象物質に対する月ごとの平均値を算出
def month_mean(material_df: pd.DataFrame):

    material_df_mean = {}

    for i in range(2010, 2022):
        for u in range(1,13):
            target = str(i) + '-' + str(u).rjust(2, '0')
            material_df_filter = material_df.filter(like = target, axis = 0)
            material_mean = np.mean(material_df_filter)

            material_df_mean[target] = material_mean
    
    return material_df_mean

#対象物質に対する月ごとの最大値を出力
def month_max(material_df: pd.DataFrame):

    material_df_max = {}

    for i in range(2018, 2020):
        for u in range(4,9):
            target = str(i) + '-' + str(u).rjust(2, '0')
            material_df_filter = material_df.filter(like = target, axis = 0)
            material_max = material_df_filter.max()

            material_df_max[target] = material_max
    
    return material_df_max

#8時間移動平均値を出力
def mean_8(material_df: pd.DataFrame):

    index_num = material_df.shape[0] + 1
    material_df_8_mean_li = []

    for i in range(index_num):
        start_point = i
        last_point = i + 7

        target_mean_data = material_df[start_point : last_point].to_numpy()
        target_mean_data_elements = target_mean_data.shape[0]

        if target_mean_data_elements == 7:
            mean = np.mean(target_mean_data)
            material_df_8_mean_li.append(mean)

        else:
            material_df = material_df.drop(index = material_df.index[start_point:])
            for u, material_df_8_mean in enumerate(material_df_8_mean_li):
                material_df.iloc[u] = material_df_8_mean

            break
    
    plt.figure()
    material_df.plot()
    plt.savefig('data/dst/pandas_iris_line.png')
    plt.close('all')

    return material_df

#データをグラフにして出力
def data_plt(x_data, y_data, x_label, y_label, title):
    plt.plot(x_data, y_data, marker="o")
    plt.xlabel = x_label
    plt.ylabel = y_label
    plt.title = title
    plt.savefig("out_data/" + title + ".png")
    plt.show()


def main(locate = '', query_items = [], save_name = '', T_flag = True):
    target_columns_point_li = [0,4,5]
    #基本出力

    #header_point,index_pointをしてすることでデータフレームのcolumns,indexにする列、行を選択
    #start_year、last_yearで開始、終了年度を選択しcsvファイルを取り込む
    df_li = csv_read(locate = locate)

    #csv_read()で作ったデータフレームの欠損値を線形保管する
    df_li = missing_value_imputation(df_li, interpolation_flag = True)

    #欠損値を補完したデータフレームに対して、query_itemsで指定したフィルタ対象に関してフィルタをかける
    df_li = extraction(df_li, query_items = query_items)

    #n年度データとなっているものを時系列データに変換
    df_li = year_correction(df_li)

    #target_columns_point_liで指定したカラムのデータを結合し出力データフレームのindexラベルに変換
    new_index_name_dict = create_label_name(df_li, target_columns_point_li)

    #上で作成したindexラベルをもとに対象物質データを対応させていくとともに一時間値データに変換し(n,1)のデータフレームに変換
    #flagをtrueにするとこの時点で作成されたデータをcsvとして出力
    df_li = new_index_date_df(df_li, new_index_name_dict, flag = False)

    #ここまで作ってきたデータフレーム全て結合させる
    out_df = join(df_li)

    out_df = df_filter(out_df, '2020/4/1/0', '2020/9/31/23')
    
    new_df = df_extraction(out_df, [2,14])

    #new_df['OX'].to_csv(dir_path + save_name + '_2_14.csv', encoding='sjis')

    out_df.drop(columns=['year', 'month', 'day', 'time'], inplace = True)

    #out_df['OX'].to_csv(dir_path + save_name + '.csv', encoding='sjis')

    #結合させたデータフレームをcsvとして出力out_data/以降を変更することで保存名を変えられる
    print(out_df)

        

"""
演算子の選択
query_itemsの0番目の値

0が完全一致
1が部分一致
2が以下
3が以上
4がquery_item[2]以上query_item[3]以下

東京 江東区大島_13108010
埼玉 鴻巣_11217010
群馬 衛生環境研究所_10201090
11421010,環境科学国際Ｃ_11421010
前橋東局_10201070
"""

dir_path = 'out_data/ホニキデータ/'
target_points = {
    "gunma": ["前橋東局_10201070"]
}

search_target = []
T_flag = True

for locate, station_li in target_points.items():
    for station in station_li:
        new_name_li = station.split('_')
        new_name = new_name_li[0]
        new_code = int(new_name_li[1])

        new_query_item = [['演算子の選択', '検索対象行名、列名', '検索条件'],
                        [0, '測定局コード', new_code]
        ]

        main(locate = locate, query_items = new_query_item, save_name = new_name, T_flag = T_flag)