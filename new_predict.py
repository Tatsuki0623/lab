import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import load_model


class ozon_predict:
    #変数の初期化
    def __init__(self,
                 predict_data_path_name: str,
                 model_path_name: str,
                 time_steps = 24,
                 features = []) -> None:
        
        predict_data_path = 'python-code/lab/input_data/learning_data/machine_learning_' + predict_data_path_name + '.csv'
        model_path = 'python-code/lab/out_data/model/ozone_prediction_lstm_model_' + model_path_name + '.h5'
        plot_path = 'python-code/lab/out_data/plot_img/new_loader/'
        features = ['OX','NOX','NMHC','TEMP'] + features
        
        self.predict_data = pd.read_csv(predict_data_path, index_col = 0, header = 0)
        self.learning_data_path_name = predict_data_path_name
        self.time_steps = time_steps
        self.model_path = model_path
        self.plot_path = plot_path
        self.features = features
        self.scarer_predict = StandardScaler()

        self.predict_data_path = None
        self.predicted = None
        self.hidden = None
        self.X_target = None
        self.y = None
        self.plot_flag = None
        self.mse = None

    #学習用データ変数の追加
    def input_scaled_data(self, input_data):
        self.X_target = input_data[0]
        self.y = input_data[1]

    # 学習に関する変数の追加
    def input_parametor(self, plot_flag):
        self.plot_flag = plot_flag
    
    # 予測値の変数追加
    def input_predict(self, mse, predict_mdoel):
        self.mse = mse
        self.predicted = predict_mdoel


    #データの前処理
    def pre_treatment(self):
        # 訓練データとテストデータの準備
        X, y = self.create_input_data()

        #LSTM学習用データに変換
        X_target = X.reshape(-1, self.time_steps, len(self.features))

        input_data = [X_target, y]
        self.input_scaled_data(input_data)

    #学習データの整形
    def create_input_data(self):
        x_data, y_data = [], [] 

        last_data = len(self.predict_data) - self.time_steps
        predict_data = scipy.stats.zscore(self.predict_data)
        OX = 0

        for i in range(last_data):
            target_point = i + self.time_steps

            x_data.append(predict_data[self.predict_data.columns[0:]][i:target_point])
            y_data.append(self.predict_data[self.predict_data.columns[OX]][target_point])

        return np.array(x_data), np.array(y_data)

    # 予測
    def predict(self):
        predict_model = load_model(filepath = self.model_path, compile = True)
        predict_model = predict_model.predict(self.X_target)
        predict_model = predict_model.reshape(predict_model.shape[0])
        mse = mean_squared_error(self.y, predict_model)

        self.input_predict(mse, predict_model)

    #予測値と実測値のプロット及びトレーニングと検証のMSEプロット
    def plot(self):
        file_name = self.plot_path + 'name=' + self.learning_data_path_name

        # 予測値と実測値のプロット
        plt.figure(figsize = (14, 7))
        plt.plot(self.y, label = 'True Ozone Concentration')
        plt.plot(self.predicted, label = 'Predicted Ozone Concentration')
        plt.title('True vs Predicted Ozone Concentration')
        plt.xlabel('Time')
        plt.ylabel('Ozone Concentration')
        plt.legend()
        plt.show()


    #起動
    def set_data(self):
        self.pre_treatment()

    def start_predict(self, plot_flag = True):
        self.input_parametor(plot_flag)
        self.predict()
        
        if self.plot_flag:
            self.plot()

        return self.mse, self.predicted


model_name = 'harumi_rayer_test_3'
predict_data_name = 'harumi_predict_2019'
a = ozon_predict(predict_data_path_name = predict_data_name, model_path_name = model_name)
a.set_data()
mse, predicted = a.start_predict()

