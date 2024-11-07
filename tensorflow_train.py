import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping


class ozon_LSTM:
    #変数の初期化
    def __init__(self,
                 learning_data_path_name: str,
                 model_path_name: str,
                 time_steps = 24,
                 features = []) -> None:
        
        learning_data_path = 'python-code/lab/input_data/learning_data/machine_learning_' + learning_data_path_name + '.csv'
        model_path = 'python-code/lab/out_data/model/ozone_prediction_lstm_model_' + model_path_name + '.h5'
        plot_path = 'python-code/lab/out_data/plot_img/new_loader/'
        
        #静的変数
        self.data = pd.read_csv(learning_data_path, index_col = None, header = 0).astype(np.float32)
        self.learning_data_path_name = learning_data_path_name
        self.time_steps = time_steps
        self.model_path = model_path
        self.plot_path = plot_path
        self.features = features
        self.scaler = StandardScaler()

        #動的変数
        self.hidden = None
        self.plot_flag = None
        self.rayer_3_flag = None
        self.history = None
        self.y_train = None
        self.y_test = None
        self.X_test_scaled = None
        self.X_train_scaled = None
        self.loss_flag = None
        self.mse = None

    #学習用データ変数の追加
    def input_scaled_data(self, input_data):
        self.y_train = input_data[0]
        self.y_test = input_data[1]
        self.X_test_scaled = input_data[2]
        self.X_train_scaled = input_data[3]

    # 学習済データ変数の追加
    def input_history(self, history, mse, y_pred):
        self.history = history
        self.mse = mse
        self.y_pred = y_pred

    # 学習に関する変数の追加
    def input_parametor(self, hidden, plot_flag, rayer_3_flag, loss_flag):
        self.hidden = hidden
        self.plot_flag = plot_flag
        self.rayer_3_flag = rayer_3_flag
        self.loss_flag = loss_flag


    #データの前処理
    def pre_treatment(self):
        # 訓練データとテストデータの準備
        X_train, X_test, y_train, y_test = self.create_input_data()
        self.data = []

        #LSTM学習用データに変換
        X_test_scaled = X_test.reshape(-1, self.time_steps, len(self.features))
        X_train_scaled = X_train.reshape(-1, self.time_steps, len(self.features))

        del X_test, X_train

        input_data = [y_train, y_test, X_test_scaled, X_train_scaled]
        self.input_scaled_data(input_data)

    def pre_treatment(self):
        # 訓練データとテストデータの準備
        X_train, X_test, y_train, y_test = self.create_input_data()
        
        # LSTM学習用データに変換
        X_test_scaled = X_test.reshape(-1, self.time_steps, len(self.features))
        X_train_scaled = X_train.reshape(-1, self.time_steps, len(self.features))

        input_data = [y_train, y_test, X_test_scaled, X_train_scaled]
        self.input_scaled_data(input_data)

    def create_input_data(self):
        x_test, y_test = [], []
        x_train, y_train = [], []

        last_data = len(self.data) - self.time_steps
        train_num = int(last_data * 0.8 + self.time_steps)
        test_num = train_num - self.time_steps

        #['OX','NO','NO2','NMHC','TEMP','HUM','sin_day','cos_day','sin_hour','cos_hour']
        
        # 訓練データとテストデータの分割
        train_data = self.data.iloc[:train_num]
        test_data = self.data.iloc[test_num:]
        
        # 訓練データを標準化
        train_data_scaled = self.scaler.fit_transform(train_data.iloc[:, :4])
        train_data = pd.concat([pd.DataFrame(train_data_scaled), train_data.iloc[:, 4:8].reset_index(drop=True)], axis=1)
        
        # テストデータを訓練データの統計量で標準化
        test_data_scaled = self.scaler.transform(test_data.iloc[:, :4])
        test_data = pd.concat([pd.DataFrame(test_data_scaled), test_data.iloc[:, 4:8].reset_index(drop=True)], axis=1)

        OX = 0

        for i in range(last_data):
            if i < test_num:
                target_point = i + self.time_steps

                x_train.append(train_data.iloc[i:target_point].values)
                y_train.append(self.data.iloc[target_point, OX])
            else:
                i -= test_num
                target_point = i + self.time_steps

                x_test.append(test_data.iloc[i:target_point].values)
                y_test.append(self.data.iloc[target_point + test_num, OX])

        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


    #学習開始
    def train(self):
        #モデルの設定
        model = Sequential()
        model.add(LSTM(self.hidden * 4, return_sequences = True, input_shape = (self.time_steps, len(self.features))))
        model.add(Dropout(0.3))
        
        if self.rayer_3_flag:
            model.add(LSTM(self.hidden, return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(self.hidden))
            model.add(Dropout(0.2))
            model.add(Dense(1))

        else:
            model.add(LSTM(self.hidden, return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(self.hidden, return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(int(self.hidden / 2), return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(int(self.hidden / 8)))
            model.add(Dropout(0.2))
            model.add(Dense(1))

        print(model.summary())
        
        if self.loss_flag: 
            loss = tf.keras.losses.MeanSquaredError()
        else:
            loss = self.weighted_mse
        
        #評価基準の設定
        model.compile(optimizer = 'adam', loss = loss)

        #学習
        #early_stopping = EarlyStopping(monitor = 'val_loss', mode = 'auto', patience = 15)
        history = model.fit(self.X_train_scaled, self.y_train, 
                            validation_split = 0.2, 
                            epochs = 70, 
                            batch_size = 64,
                            verbose = 1)
        #callbacks = [early_stopping]
        
        #テストデータを用いて予測しモデルの評価をする
        y_pred = model.predict(self.X_test_scaled)
        mse = mean_squared_error(self.y_test, y_pred)
        print(f'Mean Squared Error: {mse}')

        #生成したモデルをh5ファイルとして保存
        model.save(self.model_path)

        self.input_history(history, mse, y_pred)
    
    def weighted_mse(self, y_true, y_pred):
        threshold_high = 80.0  # 高濃度オゾンの閾値
        threshold_low = 10.0  # 低濃度オゾンの閾値
        extreme_high = 100.0  # 極端に高い濃度の閾値

        high_weight = 20.0  # 高濃度データの重み
        low_weight = 1.0  # 低濃度データの重み
        medium_weight = 1.0  # 中間濃度データの重み
        extreme_high_weight = 20.0  # 極端に高い濃度の重み

        high_condition = tf.cast(y_true >= threshold_high, dtype = tf.float32)
        low_condition = tf.cast(y_true <= threshold_low, dtype = tf.float32)
        extreme_high_condition = tf.cast(y_true >= extreme_high, dtype = tf.float32)
        medium_condition = 1 - high_condition - low_condition - extreme_high_condition

        weight = (high_condition * high_weight + low_condition * low_weight + medium_condition * medium_weight + extreme_high_condition * extreme_high_weight)
        mse = tf.square(y_true - y_pred)
        weighted_mse = tf.reduce_mean(weight * mse)

        return weighted_mse



    #予測値と実測値のプロット及びトレーニングと検証のMSEプロット
    def plot(self):
        file_name = self.plot_path + '2018name=' + self.learning_data_path_name + '_time_steps=' + str(self.time_steps) + '_hidden=' + str(self.hidden) + '_rayer_3_flag=' + str(self.rayer_3_flag)

        # 予測値と実測値のプロット
        plt.figure(figsize = (14, 7))
        plt.plot(self.y_test, label = 'True Ozone Concentration')
        plt.plot(self.y_pred, label = 'Predicted Ozone Concentration')
        plt.title('True vs Predicted Ozone Concentration')
        plt.xlabel('Time')
        plt.ylabel('Ozone Concentration')
        plt.legend()
        plt.savefig(file_name + '_mean.png')

        # トレーニングと検証のMSEプロット
        plt.figure(figsize=(14, 7))
        plt.plot(self.history.history['loss'], label='Training MSE')
        plt.plot(self.history.history['val_loss'], label='Validation MSE')
        plt.title('Training and Validation MSE')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Squared Error')
        plt.legend()
        plt.savefig(file_name + '_loss.png')


    #起動
    def set_data(self):
        self.pre_treatment()


    def start_train(self, hidden: int, plot_flag = True, rayer_3_flag = True, loss_flag = True):
        self.input_parametor(hidden, plot_flag, rayer_3_flag, loss_flag)
        self.train()
        
        if self.plot_flag:
            self.plot()

        return self.mse

learning_data_path_name = ['harumi_3']
features = ['OX','NO','TEMP','HUM','sin_day','cos_day','sin_hour','cos_hour']
model_path_name = 'harumi_rayer_5_HUM_NO_3'
hidden_li = [128]
plot_flag = True

for i in learning_data_path_name:
    a = ozon_LSTM(learning_data_path_name = i, model_path_name = model_path_name, time_steps = 24, features = features)
    a.set_data()
    
    plot_data = []
    for u in hidden_li:
        mse = a.start_train(hidden = u, rayer_3_flag = False, loss_flag = True)
        plot_data.append(mse)
    
    A_plot_path = f'python-code/lab/out_data/plot_img/new_loader/mse.png'
    plt.figure(figsize=(17, 8))
    plt.plot(hidden_li, plot_data)
    plt.savefig(A_plot_path)