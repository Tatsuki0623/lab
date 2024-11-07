import geopandas as gpd
import os
import tkinter as tk 
from tkinter import filedialog
from glob import glob

flag = False

def save_geojson():
    iDir = os.path.abspath(os.path.dirname(__file__))
    dir = filedialog.askdirectory(initialdir = iDir)

    file_li = glob(dir + '/**/*.shp')
    if file_li != []:
        for file_path in file_li:

            vector = gpd.read_file(file_path)

            file_name = file_path.split('\\')[-1]
            file_name = file_name.split('-')[2]

            save_file_path = dir + '/' + file_name + '.geojson'

            vector.to_file(save_file_path, driver = "GeoJSON")

        label5 = tk.Label(root, text ='保存が完了しました')
        label5.pack(pady = 5)
    else:
        pass

root = tk.Tk()

root.title('Save Geojson')
root.minsize(800, 500)

label3 = tk.Label(root, text = '＊選択するフォルダの中に解凍したGISデータを入れてください')
label3.pack(pady = 5)

label4 = tk.Label(root, 
                    text = '''例：選択フォルダ
                                        |- GXX-d-XX_XXXX-jgd_GML
                                                |-*.xml
                                                |-*.dbf
                                                |-*.shp
                                                |-*.shx
                                                |-*.xml
                            このような形式にしてください''')
label4.pack(pady = 5)

button1 = tk.Button(root, text = 'フォルダの選択', command = save_geojson)
button1.pack(pady = 15)

root.mainloop()