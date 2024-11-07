import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torchinfo import summary
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular
import os
import re


class OzonLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(OzonLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size

        # LSTM layer
        self.lstm1 = nn.LSTM(
            input_size, 
            hidden_size * 4,  # Corrected hidden size
            num_layers = self.num_layers, 
            batch_first = True, 
            dropout = 0.2
        )

        self.dropout1 = nn.Dropout(0.2)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 4, hidden_size // 2),  # Corrected input size
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, hidden_size // 8),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 8, output_size)
            # Add activation function if needed
        )

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out = self.fc(out[:,-1,:])
        return out

class OzonLSTMTrain:
    def __init__(self, model_path, time_steps = 24, features = [], scaler_mode = True):
        self.model_path = model_path
        self.time_steps = time_steps
        self.features = features
        self.scaler_mode = scaler_mode
        self.scaler_norm = MinMaxScaler()
        self.scaler_norm_OX = MinMaxScaler()
        self.scaler_stand = StandardScaler()
        self.scaler_stand_OX = StandardScaler()

    def pre_treatment(self, data_path):
        data = pd.read_csv(data_path, index_col = None, header = 0).astype(np.float32)
        self.data = data

        X_train, X_test, y_train, y_test = self.create_input_data()

        X_test_scaled = X_test.reshape(-1, self.time_steps, len(self.features))
        X_train_scaled = X_train.reshape(-1, self.time_steps, len(self.features))

        self.y_train = torch.tensor(y_train, dtype = torch.float)
        self.y_test = torch.tensor(y_test, dtype = torch.float)
        self.X_test_scaled = torch.tensor(X_test_scaled, dtype = torch.float)
        self.X_train_scaled = torch.tensor(X_train_scaled, dtype = torch.float)

    def create_input_data(self):
        x_test, y_test = [], []
        x_train, y_train = [], []

        last_data = len(self.data) - self.time_steps
        train_num = int(last_data * 0.8 + self.time_steps)
        test_num = train_num - self.time_steps

        OX = ['OX']
        new_features = list(set(OX + self.features))
        new_features.sort()
        self.features.sort()
        
        self.data = self.data[new_features]

        train_data = self.data.iloc[:train_num]
        test_data = self.data.iloc[test_num:]
        train_OX_data = train_data['OX'].to_numpy().reshape(-1, 1)
        test_OX_data = test_data['OX'].to_numpy().reshape(-1, 1)

        drop_OX_train_data = train_data.drop('OX', axis = 1)
        drop_OX_test_data = test_data.drop('OX', axis = 1)
        drop_OX_new_features = new_features
        drop_OX_new_features.remove('OX')

        del self.data

        if self.scaler_mode == 0:
            # 正規化
            drop_OX_train_data = pd.DataFrame(self.scaler_norm.fit_transform(drop_OX_train_data), columns = drop_OX_new_features)
            drop_OX_test_data = pd.DataFrame(self.scaler_norm.transform(drop_OX_test_data), columns = drop_OX_new_features)
            OX_train_data = pd.DataFrame(self.scaler_norm_OX.fit_transform(train_OX_data), columns = OX)
            OX_test_data = pd.DataFrame(self.scaler_norm_OX.transform(test_OX_data), columns = OX)

            train_data = pd.concat([drop_OX_train_data, OX_train_data], axis = 1)
            test_data = pd.concat([drop_OX_test_data, OX_test_data], axis = 1)
        elif self.scaler_mode == 1:
            # 標準化
            drop_OX_train_data = pd.DataFrame(self.scaler_stand.fit_transform(drop_OX_train_data), columns = drop_OX_new_features)
            drop_OX_test_data = pd.DataFrame(self.scaler_stand.transform(drop_OX_test_data), columns = drop_OX_new_features)
            OX_train_data = pd.DataFrame(self.scaler_stand_OX.fit_transform(train_OX_data), columns = OX)
            OX_test_data = pd.DataFrame(self.scaler_stand_OX.transform(test_OX_data), columns = OX)

            train_data = pd.concat([drop_OX_train_data, OX_train_data], axis = 1)
            test_data = pd.concat([drop_OX_test_data, OX_test_data], axis = 1)
        elif self.scaler_mode == 2:
            # 前処理なし
            train_data = train_data
            test_data = test_data

        OX_num = train_data.columns.get_loc('OX')

        for i in range(last_data):
            if i < test_num:
                target_point = i + self.time_steps
                x_train.append(train_data[self.features].iloc[i:target_point].values)
                y_train.append(train_data.iloc[target_point, OX_num])
            else:
                i -= test_num
                target_point = i + self.time_steps
                x_test.append(test_data[self.features].iloc[i:target_point].values)
                y_test.append(test_data.iloc[target_point, OX_num])


        return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)

    def train(self, hidden_sizes, epochs=10, batch_size=64, learning_rate=0.001, num_layers=3):
        criteria = nn.MSELoss()

        self.batch_size = batch_size
        self.criteria = criteria

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OzonLSTM(input_size=len(self.features), num_layers=num_layers, hidden_size=hidden_sizes, output_size=1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        self.X_train_scaled = self.X_train_scaled.to(device)
        self.y_train = self.y_train.to(device)
        self.X_test_scaled = self.X_test_scaled.to(device)
        self.y_test = self.y_test.to(device)

        summary(model)

        dataset = TensorDataset(self.X_train_scaled, self.y_train)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        loss_train_history = []
        test_rmse_history = []

        for epoch in range(epochs):
            epoch_loss_train = 0
            model.train()

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.view(-1)
                loss_train = torch.sqrt(self.criteria(targets, outputs))
                loss_train.backward()
                optimizer.step()
                epoch_loss_train += loss_train.item()

            average_loss_train = epoch_loss_train / len(train_loader)
            loss_train_history.append(average_loss_train)

            model.eval()
            with torch.no_grad():
                test_outputs = model(self.X_test_scaled)
                test_outputs = test_outputs.view(-1)
                
                # 予測結果を逆変換
                if self.scaler_mode == 0:  # 正規化の場合
                    test_outputs_np = test_outputs.cpu().numpy().reshape(-1, 1)
                    test_outputs_np_original = self.scaler_norm_OX.inverse_transform(test_outputs_np)
                    y_test_np_original = self.scaler_norm_OX.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1))

                elif self.scaler_mode == 1:  # 標準化の場合
                    test_outputs_np = test_outputs.cpu().numpy().reshape(-1, 1)
                    test_outputs_np_original = self.scaler_stand_OX.inverse_transform(test_outputs_np)
                    y_test_np_original = self.scaler_stand_OX.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1))

                else:  # 前処理なし
                    test_outputs_np_original = test_outputs.cpu().numpy()
                    y_test_np_original = self.y_test.cpu().numpy()

                # 元のスケールでRMSEを計算
                test_rmse = np.sqrt(((test_outputs_np_original - y_test_np_original) ** 2).mean())

            test_rmse_history.append(test_rmse)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {average_loss_train}, Test RMSE: {test_rmse}')

        torch.save(model.state_dict(), self.model_path)

        dummy_input = torch.zeros(batch_size, self.time_steps, len(self.features)).to(device)

        self.plot_loss(loss_train_history, test_rmse_history, epochs)
        rmse = self.plot_predictions(model)

        return model, rmse

    def plot_loss(self, loss_history, test_rmse_history, epochs):
        fig, ax = plt.subplots()
        ax2 = ax.twinx()
        ax.plot([i for i in range(epochs)], loss_history, label='Training Loss', c = 'r')
        ax2.plot([i for i in range(epochs)], test_rmse_history, label='Test RMSE', c = 'b')
        h, l = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.set_ylabel('Training Loss')
        ax2.set_ylabel('Test RMSE')
        plt.title('Loss Over Epochs')
        ax.legend(h + h2, l + l2)
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}.png')

    def plot_predictions(self, model):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_test_scaled = self.X_test_scaled.to(device)
        self.y_test = self.y_test.to(device)

        with torch.no_grad():
            test_outputs = model(self.X_test_scaled)
            test_outputs = test_outputs.view(-1)

            if self.scaler_mode == 0:  # 正規化の場合
                    test_outputs_np = test_outputs.cpu().numpy().reshape(-1, 1)
                    test_outputs_np_original = self.scaler_norm_OX.inverse_transform(test_outputs_np)
                    y_test_np_original = self.scaler_norm_OX.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1))

            elif self.scaler_mode == 1:  # 標準化の場合
                    test_outputs_np = test_outputs.cpu().numpy().reshape(-1, 1)
                    test_outputs_np_original = self.scaler_stand_OX.inverse_transform(test_outputs_np)
                    y_test_np_original = self.scaler_stand_OX.inverse_transform(self.y_test.cpu().numpy().reshape(-1, 1))

            else:  # 前処理なし
                    test_outputs_np_original = test_outputs.cpu().numpy()
                    y_test_np_original = self.y_test.cpu().numpy()

                # 元のスケールでRMSEを計算
        test_rmse = np.sqrt(((test_outputs_np_original - y_test_np_original) ** 2).mean())

        print(f'test case RMSE = {test_rmse}')
        plt.figure(figsize = (10, 6))
        plt.plot(y_test_np_original, label = 'True Values')
        plt.plot(test_outputs_np_original, label = 'Predicted Values')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}_訓練データ.png')

        return test_rmse

    def visualize_feature_importance_with_lime(self, model):
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.X_test_scaled = self.X_test_scaled.to(device)

        # サンプルデータの準備（各サンプルを2次元に変換）
        X_sample = self.X_test_scaled[:100].cpu().numpy()
        X_sample_mean = X_sample.mean(axis=1)

        # モデルの予測関数の定義
        def predict_fn(X):
            model.eval()
            X_tensor = torch.tensor(X, dtype=torch.float).to(device)
            X_tensor = X_tensor.unsqueeze(1).repeat(1, self.time_steps, 1)  # 2次元を3次元に変換
            with torch.no_grad():
                return model(X_tensor).cpu().numpy()

        # LIMEのセットアップ
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_sample_mean, 
            feature_names=self.features, 
            class_names=['output'], 
            mode='regression'
        )

        # 例として最初のサンプルの説明を取得
        exp = explainer.explain_instance(X_sample_mean[0], predict_fn, num_features = len(self.features))

        importance_dict = dict(exp.as_list())

        # 特徴量名をクリーンアップ
        cleaned_importance_dict = {feature.replace(' ', '').replace('>', '').replace('<', '').replace('=', ''): importance for feature, importance in importance_dict.items()}
        
        cleaned_importance_dict = self.clean(cleaned_importance_dict)
        
        # キーと値のリストを作成
        labels = list(cleaned_importance_dict.keys())
        values = list(cleaned_importance_dict.values())

        # 色のリスト（条件によって色を変える）
        colors = ['red' if value < 0 else 'blue' for value in values]

        # グラフの作成
        plt.figure(figsize = (18, 10))  # グラフのサイズを指定
        bars = plt.barh(labels, values, color=colors)  # 水平棒グラフを作成

        # 各バーの横に値を表示
        for bar, value in zip(bars, values):
            plt.text((value + 0.0001 if value >= 0 else value * -1 + 0.0006), bar.get_y() + bar.get_height()/2, 
                    f'{value:.8f}', va='center', ha='left' if value >= 0 else 'right')

        # グラフの表示
        plt.xlabel('Values')
        plt.title('Conditions and Their Corresponding Values')
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}_特徴量需要度.png')
        # 特徴量重要度を一つの表として棒グラフで表示
    
    def clean(self, clean_dict: dict):
        cleaned_dict = {}
        for key in clean_dict.keys():
            new_key = re.findall('[a-z]+', key, flags = re.IGNORECASE)
            if len(new_key) >= 2:
                new_key = ('_'.join(new_key))
            else:
                new_key = new_key[0]
            cleaned_dict[new_key] = clean_dict[key]
        
        features_df = pd.DataFrame(cleaned_dict, [num_p()])
        features_li.append(features_df)

        return cleaned_dict
        

class OzonLSTMPredictor:
    def __init__(self, model_path, time_steps = 24, features = [], scaler_mode = 0, data_path = "", hidden_sizes = 256):
        self.model_path = model_path
        self.time_steps = time_steps
        self.features = features
        self.scaler_mode= scaler_mode
        self.hidden_sizes = hidden_sizes
        self.scaler_norm = MinMaxScaler()
        self.scaler_norm_OX = MinMaxScaler()
        self.scaler_stand = StandardScaler()
        self.scaler_stand_OX = StandardScaler()
        self.linear = LinearRegression()
        self.model = self.load_model()
        self.data = pd.read_csv(data_path, index_col = None, header = 0)

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OzonLSTM(input_size = len(self.features), hidden_size = self.hidden_sizes, output_size = 1, num_layers = 3).to(device)

        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        return model

    def create_sequences(self):
        X_sequences = []
        y_sequences = []

        OX = ['OX']
        new_features = list(set(OX + self.features))
        new_features.sort()
        self.features.sort()
        
        self.data = self.data[new_features]
        OX_data = self.data['OX'].to_numpy().reshape(-1, 1)

        drop_OX_train_data = self.data.drop('OX', axis = 1)
        drop_OX_new_features = new_features
        drop_OX_new_features.remove('OX')

        if self.scaler_mode == 0:
            scaled_data = pd.DataFrame(self.scaler_norm.fit_transform(drop_OX_train_data), columns = drop_OX_new_features)
            OX_scaled_data = pd.DataFrame(self.scaler_norm_OX.fit_transform(OX_data), columns = OX)

            scaled_data = pd.concat([scaled_data, OX_scaled_data], axis = 1)
        elif self.scaler_mode == 1:
            scaled_data = pd.DataFrame(self.scaler_stand.fit_transform(drop_OX_train_data), columns = drop_OX_new_features)
            OX_scaled_data = pd.DataFrame(self.scaler_stand_OX.fit_transform(OX_data), columns = OX)

            scaled_data = pd.concat([scaled_data, OX_scaled_data], axis = 1)
        elif self.scaler_mode == 2:
            scaled_data = self.data

        last_data = len(self.data) - self.time_steps

        OX_num = scaled_data.columns.get_loc('OX')

        for i in range(last_data):
            target_point = i + self.time_steps
            X_sequences.append(scaled_data[self.features].iloc[i:target_point].values)
            y_sequences.append(scaled_data.iloc[target_point, OX_num])

        return np.array(X_sequences), np.array(y_sequences)

    def predict(self):
        criteria = nn.MSELoss()

        X_sequences, y_sequences = self.create_sequences()
        X_sequences = torch.tensor(X_sequences, dtype = torch.float)
        y_sequences = torch.tensor(y_sequences, dtype = torch.float)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_sequences = X_sequences.to(device)
        y_sequences = y_sequences.to(device)

        with torch.no_grad():
            predictions = self.model(X_sequences)

        # 逆変換処理（オゾンのみ）
        if self.scaler_mode == 0:  # 正規化の場合
            predictions_np = predictions.cpu().numpy().reshape(-1, 1)
            predictions_np_original = self.scaler_norm_OX.inverse_transform(predictions_np)
            y_sequences_np_original = self.scaler_norm_OX.inverse_transform(y_sequences.cpu().numpy().reshape(-1, 1))

        elif self.scaler_mode == 1:  # 標準化の場合
            predictions_np = predictions.cpu().numpy().reshape(-1, 1)
            predictions_np_original = self.scaler_stand_OX.inverse_transform(predictions_np)
            y_sequences_np_original = self.scaler_stand_OX.inverse_transform(y_sequences.cpu().numpy().reshape(-1, 1))
        
        else:  # 前処理なし
            predictions_np_original = predictions.cpu().numpy()
            y_sequences_np_original = y_sequences.cpu().numpy()

        # RMSEの計算は逆変換された元のスケールで行う
        rmse = np.sqrt(((predictions_np_original - y_sequences_np_original) ** 2).mean())
        
        print(f'test case RMSE = {rmse}')

        save_dict = {'予測値': predictions_np_original.reshape(-1), '実測値': y_sequences_np_original.reshape(-1)}
        save_df = pd.DataFrame(save_dict)
        self.save_df = save_df
        
        plt.figure(figsize=(16, 7))
        plt.plot(y_sequences_np_original, label='True Values')
        plt.plot(predictions_np_original, label='Predicted Values')
        plt.xlabel('Time Step')
        plt.ylabel('Ozone Level')
        plt.title('True vs Predicted Ozone Levels')
        plt.legend()
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}_新規予測.png')

        self.linear.fit(y_sequences_np_original, predictions_np_original)
        plt.figure(figsize=(16,7))
        plt.scatter(y_sequences_np_original, predictions_np_original)
        plt.plot(y_sequences_np_original, self.linear.predict(y_sequences_np_original), linestyle="solid", c = 'r')
        plt.xlabel('True')
        plt.ylabel('predict')
        plt.title('True vs Predicted Ozone Levels')
        plt.legend([
                f'y= {self.linear.coef_}x + {self.linear.intercept_}',
                f'R^2: {self.linear.score(y_sequences_np_original, predictions_np_original)}'
        ])
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}_散布図.png')

        check_li = self.high_concent_check()
        check_li.append(self.lag_check())

        return rmse, check_li
    
    def high_concent_check(self):
        check_df = self.save_df
        check_df = check_df.query('実測値 >= 100')

        check_df_total = check_df.shape[0]
        check_df_true =  check_df.query('予測値 >= 100').shape[0]
        check_par = int((((check_df_true / check_df_total) * 100) * 1000)) / 1000
        check_rmse = mean_squared_error(check_df['実測値'], check_df['予測値'], squared = False)

        check_li = [check_df_total, check_df_true, check_par, check_rmse]

        return check_li
    
    def lag_check(self):
        check_df = self.save_df
        check_df['予測値_lug'] = check_df['予測値'].shift(-1)
        check_df = check_df.iloc[:-1]
        lag_rmse = mean_squared_error(check_df['実測値'], check_df['予測値_lug'], squared = False)
        self.save_df.to_csv(f'python-code/lab/out_data/results/out_csv/{name}_timesteps={self.time_steps}_scalermode={self.scaler_mode}_list={list_join}.csv')

        return lag_rmse




def training(time_step, feature):
    # トレーニング
    trainer = OzonLSTMTrain(model_path = model_path, time_steps = time_step, features = feature, scaler_mode = scaler_mode)
    trainer.pre_treatment(data_train_path)
    trained_model, rmse = trainer.train(hidden_sizes, epochs = epoch_num)
    trainer.visualize_feature_importance_with_lime(trained_model)

    return rmse

def predicting(time_step, feature):
    # 予測
    predicter = OzonLSTMPredictor(model_path = model_path, time_steps = time_step, features = feature, scaler_mode = 0, data_path = data_test_path, hidden_sizes = hidden_sizes)
    rmse, check_par = predicter.predict()

    return rmse, check_par

def main(main_mode: int, time_step: int, feature: list):
    if main_mode == 0:
        traing_rmse = training(time_step = time_step, feature = feature)
        predict_rmse, check_li = predicting(time_step = time_step, feature = feature)

        return traing_rmse, predict_rmse, check_li
    
    elif main_mode == 1:
        traing_rmse = training(time_step = time_step, feature = feature)
        dumy = 0
        dumy2 = 0

        return traing_rmse, dumy, dumy2
    
    elif main_mode == 2:
        predict_rmse, check_li = predicting(time_step = time_step, feature = feature)
        dumy = 0

        return dumy, predict_rmse, check_li

def features_plot(cleaned_importance_dict):
        labels = list(cleaned_importance_dict.keys())
        values = list(cleaned_importance_dict.values())

        # 色のリスト（条件によって色を変える）
        colors = ['red' if value < 0 else 'blue' for value in values]

        # グラフの作成
        plt.figure(figsize = (18, 10))  # グラフのサイズを指定
        bars = plt.barh(labels, values, color=colors)  # 水平棒グラフを作成

        # 各バーの横に値を表示
        for bar, value in zip(bars, values):
            plt.text((value + 0.0001 if value >= 0 else value * -1 + 0.0006), bar.get_y() + bar.get_height()/2, 
                    f'{value:.8f}', va='center', ha='left' if value >= 0 else 'right')

        # グラフの表示
        plt.xlabel('Values')
        plt.title('Conditions and Their Corresponding Values')
        plt.savefig(f'python-code/lab/out_data/results/image/{name}_timesteps={time_steps[0]}_scalermode={scaler_mode}_list={list_join}_特徴量需要度.png')
        # 特徴量重要度を一つの表として棒グラフで表示

def num_p():
    global num
    num += 1
    return num

def abs_dict(target_dict: dict):
    for key in target_dict.keys():
        target_dict[key] = abs(target_dict[key])
    return target_dict

def del_features_li(target_dict: dict):
    del_li = []
    for key in target_dict.keys():
        if target_dict[key] < 0:
            del_li.append(target_dict[key])
    
    return del_li

    


name = 'harumi'

'''
0: 正規化
1: 標準化
2: 前処理なし 
'''

num = 0
main_mode = 0
scaler_mode = 0
flag = False
time_steps = [24]
hidden_sizes = 256
epoch_num = 60
features = ['CH4','HUM','NOX','OX','PM25','SO2','SPM','TEMP','THC','WD','WS','sin_day','cos_day','sin_hour','cos_hour']
features_li = []
'''
2   まで   0    ['CH4', 'HUM', 'NMHC', 'NO', 'No2', 'NOX', 'OX', 'THC', 'WS', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
3   から   1
7   から   2
9   から   3
10  から   4
14  から   5
16  から   6
'CH4','HUM','NMHC','NO','NO2','NOX','OX','PM25','SO2','SPM','TEMP','THC','WD','WS','sin_day','cos_day','sin_hour','cos_hour'


['CH4', 'HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'THC', 'WS', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
['HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'THC', 'WS', 'sin_day', 'cos_day','sin_hour', 'cos_hour'], 
['HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'WS', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
['NMHC', 'NO', 'NO2', 'NOX', 'OX', 'WS', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
['NMHC', 'NO', 'NO2', 'NOX', 'OX', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
['NO', 'NOX', 'OX', 'sin_day', 'cos_day','sin_hour', 'cos_hour'],
['NO', 'OX', 'sin_day', 'cos_day','sin_hour', 'cos_hour']

'NO2', 'TEMP', 'sin_day'

'CH4', 'HUM', 'NMHC', 'NOX', 'OX', 'PM25', 'SO2', 'WD', 'WS', 'cos_day', 'sin_hour'

'''


out_li = []
while len(features) > 3:
    u = features
    for s in range(10):
        print(s)
        i = time_steps[0]
        u.sort()
        list_join = ('：'.join(u))
        data_train_path = f'python-code/lab/input_data/learning_data/machine_learning_{name}_3.csv'
        data_test_path = f'python-code/lab/input_data/learning_data/machine_learning_{name}_predict_2019.csv'
        model_path = f'python-code/lab/out_data/results/model/{name}_timesteps={i}_scalermode={scaler_mode}_list={list_join}.pth'
        traing_rmse, predict_rmse, check_li = main(main_mode, time_step = i, feature = u)
        lug_strength = check_li[4] - predict_rmse
        out_dict = {'新規予測RMSE': predict_rmse, '高濃度出現回数': check_li[0], '高濃度追跡': check_li[1], '高濃度追跡率': check_li[2], '高濃度RMSE': check_li[3]}
        out_df = pd.DataFrame(out_dict, index = [s])
        out_li.append(out_df)    
        os.remove(model_path)

    out_df = pd.concat(out_li)
    out_df = out_df.mean().to_list()
    out_dict = {'新規予測RMSE': out_df[0], '高濃度出現回数': out_df[1], '高濃度追跡': out_df[2], '高濃度追跡率': out_df[3], '高濃度RMSE': out_df[4]}
    out_df = pd.DataFrame(out_dict, index = [0])
    out_df.to_csv(f'python-code/lab/out_data/results/out_csv/{name}_timesteps={i}_scalermode={scaler_mode}_list={list_join}_出力結果.csv', index = False)

    out_features = pd.concat(features_li)
    out_features = out_features.mean().to_dict()
    features_plot(out_features)
    if flag:
        features_sorted = abs_dict(out_features)
        features_sorted = sorted(features_sorted.items(), key = lambda x:x[1], reverse = True)
        last_key = next(reversed(features_sorted), None)
        last_key = last_key[0]
        if last_key == 'No':
            last_key = 'No2'
        elif last_key == 'PM':
            last_key = 'PM25'
        elif last_key == 'CH':
            last_key = 'CH4'
        elif last_key == 'SO':
            last_key = 'SO2'
        features.remove(last_key)
    
    else:
        del_features = del_features_li(out_features)
        
        if len(del_features) != 0:
            for del_f in del_features:
                features.remove(del_f)
        else:
            features = []
            
        
