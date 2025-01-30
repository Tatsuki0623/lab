import pandas as pd
import numpy as np
import datetime
from glob import glob

def create(path, save_name, flags, time_step, year):
    target_df = pd.read_csv(path, index_col = 0, header = 0)
    target_df.index = pd.to_datetime(target_df.index)
    target_df.index.name = 'date'
    target_df.dropna(how = 'all', axis = 1, inplace = True)
    s_time = str(24 - time_step).zfill(2)
    print(save_name)
    print(target_df.columns.to_list())

    for i in flags:
        if i:
            s_dt = datetime.datetime.strptime(f'{year}-03-31 {s_time}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            e_dt = datetime.datetime.strptime(f'{year + 1}-03-31 23:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query_df = target_df.query(f'\'{s_dt}\' <= date <= \'{e_dt}\'')
            Ox_query_df = query_df.rename(columns={'NMHC': 'NMHC_lag_0'})
            Ox_df_0 = Ox_query_df['NMHC_lag_0'].iloc[time_step:]
            del Ox_query_df
            name_li = query_df.columns.to_list()
            cat_li = []
            cat_li.append(Ox_df_0)

            for i in range(1,3):
                s_lag_dt = datetime.datetime.strptime(f'{year}-03-31 {s_time}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                e_lag_dt = datetime.datetime.strptime(f'{year + 1}-04-01 0{str(i - 1)}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                Ox_df = target_df.rename(columns = {'NMHC': f'NMHC_lag_{i}'}).query(f'\'{s_lag_dt}\' <= date <= \'{e_lag_dt}\'').shift(-i)[f'NMHC_lag_{i}'][time_step:-i]
                cat_li.append(Ox_df)

            new_data_li = []
            for name in name_li:
                target_material_data = query_df[name].to_list()
                new_dict = {}
                for i in range(time_step):
                    lag = time_step - i
                    new_dict[name + '_' + str(lag).zfill(2)] = target_material_data[i : -lag]
                new_dict = dict(sorted(new_dict.items()))
                new_dict = pd.DataFrame(new_dict, index = query_df.index[time_step:])
                new_data_li.append(new_dict)
            learning_df = pd.concat(new_data_li, axis = 1)
            cat_li.append(learning_df)

            learning_df = pd.concat(cat_li, axis = 1)
            learning_df.to_csv(f'out_data/test_data/{save_name[0]}_{str(time_step)}.csv')
        else:
            s_dt = datetime.datetime.strptime(f'{year + 1}-03-31 {s_time}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            e_dt = datetime.datetime.strptime(f'{year + 2}-03-31 23:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
            query_df = target_df.query(f'\'{s_dt}\' <= date <= \'{e_dt}\'')
            name_li = query_df.columns.to_list()
            Ox_query_df = query_df.rename(columns={'NMHC': 'NMHC_lag_0'})
            Ox_df_0 = Ox_query_df['NMHC_lag_0'].iloc[time_step:]
            del Ox_query_df
            cat_li = []
            cat_li.append(Ox_df_0)

            for i in range(1,3):
                s_lag_dt = datetime.datetime.strptime(f'{year + 1}-03-31 {s_time}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                e_lag_dt = datetime.datetime.strptime(f'{year + 2}-04-01 0{str(i - 1)}:00:00', '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
                Ox_df = target_df.rename(columns = {'NMHC': f'NMHC_lag_{i}'}).query(f'\'{s_lag_dt}\' <= date <= \'{e_lag_dt}\'').shift(-i)[f'NMHC_lag_{i}'][time_step:-i]
                cat_li.append(Ox_df)

            new_data_li = []
            for name in name_li:
                target_material_data = query_df[name].to_list()
                new_dict = {}
                for i in range(time_step):
                    lag = time_step - i
                    new_dict[name + '_' + str(lag).zfill(2)] = target_material_data[i : -lag]
                new_dict = dict(sorted(new_dict.items()))
                new_dict = pd.DataFrame(new_dict, index = query_df.index[time_step:])
                new_data_li.append(new_dict)
            predict_df = pd.concat(new_data_li, axis = 1)
            cat_li.append(predict_df)

            predict_df = pd.concat(cat_li, axis = 1)
            predict_df.to_csv(f'out_data/test_data/{save_name[1]}_{str(time_step)}.csv')


flags = [True, False]
template_name = ['_NMHC_learning', '_NMHC_predict']

dir_name = 'out_data/test_data/target_dir/'
path_li = glob(dir_name + '*.csv')
for path in path_li:
    state_name = path.split('\\')[-1].split('.')[0] # 測定局名
    save_name = [state_name + nn for nn in template_name]
    time_step = 24  # 何時間分のデータを取り込むか
    year = 2018 # 学習データの開始年、これに+2年したものが検証データとなる

    create(path, save_name, flags, time_step, year)

