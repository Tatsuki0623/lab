import tkinter as tk
from tkinter import filedialog
import pandas as pd
import csv
import sys
import os

class filter:
#インスタンス化時の処理。主に変数への代入
    def __init__(self, target_index_name: str, target_colume_name: str, save_csv_name: str, high_concent: int, low_concent: int, sector_low:int , sector_high: int , mode: int):

        #filedialogでファイルを選択
        idir = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))                              #このファイルが実行されているパスの取得
        filetype = [("csvファイル","*.csv")]
        target_csv_path = tk.filedialog.askopenfilename(filetypes = filetype, initialdir = idir)        #filedialogの起動

        #選択しなかったときに強制終了
        if target_csv_path == '':
            sys.exit()


        self.mode = mode                                    #探索するデータの変更
        self.target_index_name = target_index_name          #探索対象データのラベル名称
        self.target_colume_name = target_colume_name        #インデックスに設定するカラムラベルの名称
        self.target_csv_path = target_csv_path              #csvファイルのパス
        self.save_csv_name = save_csv_name                  #保存するcsvファイルの名称
        self.high_concent = high_concent                    #入力値以上の出力
        self.low_concent = low_concent                      #入力値以下の出力
        self.sector_low = sector_low                        #範囲探索の最小値
        self.sector_high = sector_high                      #範囲探索の最大値


        self.df = None                                      #入力csvのpandasデータ
        self.date_data = None                               #日付データ
        self.start_index_num = None                         #日付探索の開始日インデックス番号
        self.finish_index_num = None                        #日付探索の終了日インデックス番号                        
        self.to_csv_li = None                               #csv出力データの一時保管リスト
        self.save_df = None                                 #csv出力データのpandasデータ
    
#csvファイルのデータをパンダスに変換
    def read_data(self):
        
        df_data = []
        df_data_colum_name = []

        #UTF-8でエンコードするためcsvファイルをのエンコードを合わせる
        with open(self.target_csv_path, encoding = 'utf-16', newline = '') as f:
            csvreader = csv.reader(f)
            for index, row in enumerate(csvreader):
                if index == 0:                                              #一行目をカラム名にするために分けてリストに
                    df_data_colum_name.append(row)
                else:
                    df_data.append(row)
        
        df = pd.DataFrame(df_data, columns = df_data_colum_name[0])         
        df.set_index(self.target_colume_name, inplace = True)               #特定の列をインデックスとして登録
        df = df.astype('float32')                                           #floatとして保存

        self.df = df                                                        #データフレームをselfに格納

#日付データをリストで取得　YYYY/MM/DD 形式
    def index_split(self):
        
        #空白+0時にすることで開始値である　'YYYY/MM/DD 0時' を取得できる
        index_filter = self.df.filter(like =' 0時', axis = 0)

        date_data = []
        index_name = index_filter.index         #全インデックスラベルを取得

        for i in index_name:
            split_li = i.split(' ')
            split_str = split_li[0]             #空白でスプリットし　YYYY/MM/DD　を取得
            date_data.append(split_str)
        
        print(date_data)                        #コンソールに日付を出力しコピペで開始終了を入力できるようにする

        self.date_data = date_data              #日付データをselfに格納

#日付データのインデックス番号の取得
    def date_section_select(self):
        
        start_date = input('開始日：')
        start_index_num = self.date_data.index(start_date)      #開始日をインデックスで取得

        finish_date = input('終了日：')
        finish_index_num = self.date_data.index(finish_date)    #終了日をインデックスで取得
        
        self.start_index_num = start_index_num                  #開始日のインデックス番号をselfに格納
        self.finish_index_num = finish_index_num                #終了日のインデックス番号をselfに格納

#データの探索
    def filter(self):
        
        #modeを切り替えることによって探索対象を変える
        if self.mode == 0:                                                          #日の最大値の探索
            
            to_csv_li = []
            
            for i in range(self.start_index_num, self.finish_index_num + 1):
                
                save_li = []
                target_date = self.date_data[i]

                df_filter = self.df.filter(like = target_date, axis = 0)            #indexに対してのフィルタ
                df_filter = df_filter.filter(items = [self.target_index_name])      #columeに対してのフィルタ

                df_filter_max = self.df_max(df_filter)                                        #探索開始

                save_li.append(target_date)                                         #探索対象日付
                save_li.append(df_filter_max)                                                 #探索対象データ

                to_csv_li.append(save_li)                                           #['YYYY/MM/DD','最大値']で格納
            
            self.to_csv_li = to_csv_li                                              #探索データリストをselfに格納
        
        elif self.mode == 1:                                                        #日の最小値の探索
            
            to_csv_li = []

            for i in range(self.start_index_num, self.finish_index_num + 1):
                
                save_li = []
                target_date = self.date_data[i]

                df_filter = self.df.filter(like = target_date, axis = 0)            #indexに対してのフィルタ
                df_filter = df_filter.filter(items = [self.target_index_name])      #columeに対してのフィルタ

                df_filter_min = self.df_min(df_filter)                              #探索開始

                save_li.append(target_date)                                         #探索対象日付
                save_li.append(df_filter_min)                                       #探索対象データ

                to_csv_li.append(save_li)                                           #['YYYY/MM/DD','最小値']で格納
            
            self.to_csv_li = to_csv_li                                              #探索データリストをselfに格納
        
        elif self.mode == 2:                                                        #入力値以上の値の探索

            df_filter = self.df
            df_filter = df_filter.filter(items = [self.target_index_name])          #columeに対してのフィルタ

            df_filter_low_sector = self.df_low_concent(df_filter)                   #探索開始
            
            self.save_df = df_filter_low_sector                                     #探索データフレームをselfに格納
        
        elif self.mode == 3:                                                        #入力値以下の値の検索

            df_filter = self.df
            df_filter = df_filter.filter(items = [self.target_index_name])          #columeに対してのフィルタ

            df_filter_high_sector = self.df_high_concent(df_filter)                   #探索開始
            
            self.save_df = df_filter_high_sector                                     #探索データフレームをselfに格納

        elif self.mode == 4:                                                        #入力値以上、入力値以下の探索

            df_filter = self.df
            df_filter = df_filter.filter(items = [self.target_index_name])          #columeに対してのフィルタ

            df_filter_sector = self.df_sector_concent(df_filter)                    #探索開始
            
            self.save_df = df_filter_sector                                         #探索データフレームをselfに格納

#最大値の探索
    def df_max(self, df_filter):
        df_filter_max = df_filter[self.target_index_name].max()
        return df_filter_max

#最小値の探索
    def df_min(self, df_filter):
        df_filter_min = df_filter[self.target_index_name].min()
        return df_filter_min

#入力値以下の探索
    def df_high_concent(self, df_filter):
        df_filter_low_sector = df_filter[df_filter[self.target_index_name] > self.low_concent]
        return df_filter_low_sector

#入力値以上の探索
    def df_low_concent(self, df_filter):
        df_filter_high_sector = df_filter[df_filter[self.target_index_name] < self.high_concent]
        return df_filter_high_sector

#入力値以上、入力値以下の探索
    def df_sector_concent(self, df_filter):
        df_filter_sector = df_filter[self.sector_low < df_filter[self.target_index_name]]
        df_filter_sector = df_filter_sector[df_filter[self.target_index_name] < self.sector_high]
        return df_filter_sector

#最大、最小の探索データをcsvとして出力
    def to_csv_polar(self):
        save_df = pd.DataFrame(self.to_csv_li)
        save_df.to_csv(self.save_csv_name + '.csv', header = False , index = False)

#範囲探索のデータをcsvとして出力
    def to_csv_sector(self):
        self.save_df.to_csv(self.save_csv_name + '.csv', header = False)

#プログラム本体、インスタンス化後の処理を制御
    def main(self):
        
        self.read_data()
        if self.mode == 0 or self.mode == 1:                        #最大、最小探索時、こちらの処理が走る
            self.index_split()
            self.date_section_select()
            self.filter()
            self.to_csv_polar()
        elif self.mode == 2 or self.mode == 3 or self.mode == 4:    #範囲探索時、こちらの処理が走る。（範囲探索時、index_split・date_section_selectの処理は余計なので除去）
            self.filter()
            self.to_csv_sector()
        

'''
------------------下記の値を変えることで探索対象や方法を切り替える-------------------



下の3つに関して「''」で文字を囲むこと

save_csv_name          :保存（出力する）csvファイルの名称(拡張子なし)

target_index_name      :探索対象のカラムラベル（部分一致）      
target_colume_name     :インデックスに指定するカラムラベル（完全一致）

下の5つに関しては数字を直接入力すること

high_concent           :入力値未満の値を探索する
low_concent            :入力値より大きいの値を探索する

sector_low             :範囲探索の最小値(入力値を含まない)
sector_high            :範囲探索の最大値(入力値を含まない)

mode                   :modeの切り替え

                        mode番号:機能
                        
                        0:最大値の探索
                        1:最小値の探索
                        2:X < input
                        3:X > input
                        4:input_l < X < input_h

'''

save_csv_name = '60_num'

target_index_name = 'その他8時間平均値'
target_colume_name = '観測日時'

high_concent = 10
low_concent = 50

sector_low = 50
sector_high = 60

mode = 0

'-----------------------------------------------------ここまで----------------------------------------------------'

f = filter(target_index_name, target_colume_name, save_csv_name, high_concent, low_concent, sector_low, sector_high, mode)
f.main()
