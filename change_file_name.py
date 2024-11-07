import os
import glob as gb

def rename_file(old_name, new_name):
    try:
        os.rename(old_name, path + new_name)
        print(f"ファイル名 {old_name} を {new_name} に変更しました。")
    except FileNotFoundError:
        print("指定されたファイルが見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

# ファイル名を変更したいファイルのパスを設定


target = {'01':'SO2','02':'NO','03':'NO2','04':'NOX','05':'CO','06':'OX','07':'NMHC','08':'CH4','09':'THC(THCP,THCM)','10':'SPM(SPMB,SPMP)','11':'SP','12':'PM2.5','21':'WD','22':'WS','23':'TEMP','24':'HUM','25':'SUN','26':'RAIN','27':'UV','29':'NETR','31':'CAR','41':'CO2','42':'オゾン','43':'塩化水素','44':'フッ化水素','45':'硫化水素','51':'PM25'}
flag = [False,True]

for n in flag:
    for u in range(2013, 2014):

        path = "python-code/lab/input_data/kankyou_date/tiba/12千葉/" + str(u) + "/"
        result = gb.glob(path + "*")

        for i in result:

            file_path = i.split("_")[-1]
            file_name = file_path.split(".")[0]

            # ファイル名を変更したいファイルのパスを設定
            if n:
                file_path = i.split("/")[-1]
                file_name_y = file_path.split("\\")[-1]
                file_name = file_name_y.split(".")[0]
                old_file_name = i
                new_file_name = file_path.split("\\")[0] + '_' + target[file_name] + '.csv'

            else:
                file_path = i.split("_")[-1]
                file_name = file_path.split(".")[0]
                old_file_name = i
                new_file_name = file_name + '.csv'


            # ファイル名を変更する関数を呼び出し
            rename_file(old_file_name, new_file_name)
