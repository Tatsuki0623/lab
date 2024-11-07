import pandas as pd
import numpy as np
from datetime import datetime

def create(path, name = [], date_flag = True):
    for i,u in enumerate(name):
        a = pd.read_csv(path, index_col = 0, header = 0)

        if i == 0:
            l_start_date = 2019
            l_last_date = l_start_date
        else:
            l_start_date = 2016
            l_last_date = l_start_date + 2

        b = a.loc[str(l_start_date) + '-01-01 00:00:00': str(l_last_date) + '-12-31 23:00:00',:]
        date_index = b.index
        
        date_data = pd.to_datetime(date_index, format='%Y-%m-%d %H:%M:%S')
        date_df = pd.DataFrame(date_data, index = None, columns = ['datetime'])

        if date_flag:

            date_df['day_of_year'] = date_df['datetime'].dt.dayofyear
            date_df['sin_day'] = np.sin((2 * np.pi * date_df['day_of_year']) / 365)
            date_df['cos_day'] = np.cos((2 * np.pi * date_df['day_of_year']) / 365)

            # 時間データを周期的な特徴量に変換 
            date_df['hour'] = date_df['datetime'].dt.hour
            date_df['sin_hour'] = np.sin((2 * np.pi * date_df['hour']) / 24)
            date_df['cos_hour'] = np.cos((2 * np.pi * date_df['hour']) / 24) 

            date_df = date_df.drop(['day_of_year', 'hour', 'datetime'], axis = 1)
            date_df = pd.DataFrame(date_df.values, index = date_index, columns = date_df.columns)

            b = pd.concat([b, date_df], axis = 1)

        else:
            date_li = []
            for date in date_df['datetime']:
                month = date.month
                day = date.day
                hour = date.hour

                append_date = {'month': month, 'day': day, 'hour': hour}
                date_li.append(append_date)
            
            date_df = pd.DataFrame(date_li, index = date_index)
            b = pd.concat([b, date_df], axis = 1)

        b = b.dropna(how = 'all', axis = 1)

        b.to_csv('python-code/lab/input_data/learning_data/machine_learning_' + u + '.csv', index = False) 

path = 'python-code/lab/out_data/verification/harumi.csv'
template_name = ['_predict_2019', '_3']
state_name = 'harumi'
name = [state_name + nn for nn in template_name]
date_flag = True

create(path, name, date_flag)