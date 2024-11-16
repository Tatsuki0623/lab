import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
import shap
import os
from concurrent.futures import ProcessPoolExecutor

class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, transform = None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        if self.transform:
          out_data = self.transform(self.data)[0][idx]
          out_label = self.label[idx]
        else:
          out_data = self.data[idx]
          out_label =  self.label[idx]

        return out_data, out_label

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()

        # 各クラスのインスタンス（入出力サイズなどの設定）
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256,128)
        self.fc5 = nn.Linear(128, output_size)


    def forward(self, x):
        # 順伝播の設定（インスタンスしたクラスの特殊メソッド(__call__)を実行）
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.relu(x)
        x = self.fc4(x)
        x = torch.relu(x)

        x = self.fc5(x)
        return x

class train():
    def __init__(self, trian_data_path, test_data_path, features, lag):
        self.num_batch = 32
        self.output_num = 1
        self.model_flag = False
        if not isinstance(lag, int) or not (0 <= lag <= 2):
            raise ValueError("引数は0以上2以下の整数でなければなりません。")

        # trainデータの作成
        train_data = pd.read_csv(trian_data_path, index_col = 0, header = 0)
        self.features = features
        train_data_label = train_data[f'OX_lag_{lag}'].to_numpy()
        self.train_data = train_data.loc[:, features]
        self.input_num = self.train_data.shape[1]
        train_data_set = MyDataset(torch.from_numpy(self.train_data.to_numpy()).float(), torch.from_numpy(train_data_label).long())

        # testデータの作成
        test_data = pd.read_csv(test_data_path, index_col = 0, header = 0)
        self.test_data_label = test_data[f'OX_lag_{lag}'].to_numpy()
        self.test_data = test_data.loc[:, features]
        self.index = test_data.index.to_list()
        test_date_set = MyDataset(torch.from_numpy(self.test_data.to_numpy()).float(), torch.from_numpy(self.test_data_label).long())

        self.train_dataloader = torch.utils.data.DataLoader(
            train_data_set,
            batch_size = self.num_batch,
            shuffle = True)
        self.test_data_loader = torch.utils.data.DataLoader(
            test_date_set,
            batch_size = 1,
            shuffle = False)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {self.device} device")

    def start_train(self, num_epochs = 70):
        self.learning_rate = 0.001
        #----------------------------------------------------------
        # ニューラルネットワークの生成
        self.model = Net(self.input_num, self.output_num).to(self.device)

        #----------------------------------------------------------
        # 損失関数の設定
        self.criterion = nn.MSELoss()

        #----------------------------------------------------------
        # 最適化手法の設定
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)

        #----------------------------------------------------------
        # 学習
        self.model.train()  # モデルを訓練モードにする

        for epoch in range(num_epochs): # 学習を繰り返し行う
            loss_sum = 0

            for inputs, labels in self.train_dataloader:

                # GPUが使えるならGPUにデータを送る
                inputs = inputs
                labels = labels
                # optimizerを初期化
                optimizer.zero_grad()

                # ニューラルネットワークの処理を行う
                inputs = inputs.view(-1, self.input_num).to(self.device)
                labels = labels.view(-1, self.output_num).float().to(self.device)
                outputs = self.model(inputs)
                # 損失(出力とラベルとの誤差)の計算
                loss = self.criterion(outputs, labels)

                loss_sum += loss

                # 勾配の計算
                loss.backward()

                # 重みの更新
                optimizer.step()

            # 学習状況の表示
            print(f"{state_name}_Epoch: {epoch+1}/{num_epochs}, Loss: {np.sqrt(loss_sum.item() / len(self.train_dataloader))}")

            # モデルの重みの保存
            torch.save(self.model.state_dict(), save_dir + 'model_weights.pth')
        self.model_flag = True

    def start_predict(self, model_path = None):
        if self.model_flag:
            pass
        else:
            self.model = Net(self.input_num, self.output_num).to(self.device)
            self.model.load_state_dict(torch.load(model_path))
            self.criterion = nn.MSELoss()

        self.model.eval()

        y=[]
        all_data_num = 0
        loss_sum = 0
        with torch.no_grad():
            for inputs, labels in self.test_data_loader:

                # GPUが使えるならGPUにデータを送る
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # ニューラルネットワークの処理を行う
                inputs = inputs.view(-1, self.input_num)
                labels = labels.view(-1, self.output_num).float()
                outputs = self.model(inputs)
                #print(outputs)
                y.append(outputs.to('cpu').detach().numpy().copy())
                all_data_num += 1

                # 損失(出力とラベルとの誤差)の計算
                loss_sum += self.criterion(outputs, labels)

            print(f"RMSE: {np.sqrt(loss_sum.to('cpu').detach().numpy() / all_data_num)}")
        out = pd.DataFrame(np.array(y).reshape(-1, 1), index = self.index, columns = ['predict'])
        true_ox = pd.DataFrame(self.test_data_label, index = self.index, columns = ['obs'])
        out = pd.concat([out, true_ox], axis = 1)
        if training_num == max_training_num -1:
            self.plot(out)
        self.high_concent(out)
        self.shap_high_feature_importance()
        self.shap_feature_importance()
        out.to_csv(save_dir + 'out.csv')

        return self.out_df

    def plot(self, out: pd.DataFrame):
        plt.figure(figsize = (16, 7))
        plt.plot(out['obs'], label = 'True Values')
        plt.plot(out['predict'], label = 'Predicted Values')
        plt.xlabel('Time Step')
        plt.ylabel('Ozone Level')
        plt.title('True vs Predicted Ozone Levels')
        plt.legend()
        plt.savefig(save_dir + '新規予測.png')
        print('plot終了')

    def high_concent(self, out: pd.DataFrame):
        rmse = root_mean_squared_error(out['obs'], out['predict'])
        high_out = out.query('obs >= 80')
        high_concent_day = high_out.index.to_list()
        high_out_len = len(high_out)
        high_rmse = root_mean_squared_error(high_out['obs'], high_out['predict'])
        high_out_p = len(high_out.query('predict >= 80'))

        high_concent_dict = {'RMSE': rmse,
                           '高濃度出現回数': high_out_len,
                           '高濃度追跡': high_out_p,
                           '高濃度追跡率': (high_out_p / high_out_len) * 100,
                           '高濃度RMSE': high_rmse
                           }

        self.out_df = pd.DataFrame(high_concent_dict, index = [0])
        self.high_concent_day = high_concent_day

    def shap_feature_importance(self, sample_size=1500):
        self.model.eval()

        X_test_sampled = self.test_data.drop(self.high_concent_day).sample(n = sample_size, random_state = 42)
        index = X_test_sampled.index.to_list()
        X_train_shap = torch.from_numpy(X_test_sampled.to_numpy()).float().to(self.device)

        explainer = shap.DeepExplainer(self.model.to(self.device), X_train_shap)
        shap_values = explainer.shap_values(X_train_shap, check_additivity=False)
        shap_values = np.array(shap_values).reshape(sample_size, len(self.features))

        # DataFrameに変換（特徴量ごとのSHAP値をカラムにする）
        shap_df = pd.DataFrame(shap_values,index = index, columns=[f'SHAP_{name}' for name in self.features])

        try:
            old_shap_df = pd.read_csv(save_dir + 'shap_explanation.csv', header = 0, index_col = 0)
            shap_df += old_shap_df
            if training_num == max_training_num - 1:
                shap_df /= max_training_num
            shap_df.to_csv(save_dir + 'shap_explanation.csv')
            print("CSVに保存完了〜！💅✨")
        except FileNotFoundError:
            shap_df.to_csv(save_dir + 'shap_explanation.csv')

    def shap_high_feature_importance(self):
        self.model.eval()

        X_test_sampled = self.test_data.loc[self.high_concent_day, :]
        index = X_test_sampled.index.to_list()
        X_train_shap = torch.from_numpy(X_test_sampled.to_numpy()).float().to(self.device)

        explainer = shap.DeepExplainer(self.model.to(self.device), X_train_shap)
        shap_values = explainer.shap_values(X_train_shap, check_additivity=False)
        shap_values = np.array(shap_values).reshape(len(self.high_concent_day), len(self.features))

        # DataFrameに変換（特徴量ごとのSHAP値をカラムにする）
        shap_df = pd.DataFrame(shap_values,index = index, columns=[f'SHAP_{name}' for name in self.features])
        try:
            old_shap_df = pd.read_csv(save_dir + 'shap_high_explanation.csv', header = 0, index_col = 0)
            shap_df += old_shap_df
            if training_num == max_training_num - 1:
                shap_df /= max_training_num
            shap_df.to_csv(save_dir + 'shap_high_explanation.csv')
            print("CSVに保存完了〜！💅✨")
        except FileNotFoundError:
            shap_df.to_csv(save_dir + 'shap_high_explanation.csv')

# ['CH4', 'HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'PM25', 'SO2', 'SPM', 'TEMP', 'THC', 'WD', 'WS'] # 晴海
# ["CH4","CO","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 東秩父　**
# ["CH4","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 幸手　**
# ["CH4","HUM","NMHC","NO","NO2","NOX","OX","PM25","SPM","TEMP","THC","WD","WS"]# 江戸川区南葛西　**
# ["CH4","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 鴻巣　**
# ["CH4","CO","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 世田谷区世田谷　**
# ["CH4","CO","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 青梅市東青梅　**
# ["CH4","CO","HUM","NMHC","NO","NO2","NOX","OX","PM25","SO2","SPM","TEMP","THC","WD","WS"]# 多摩市愛宕　**
# ['CH4', 'HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'PM25', 'SO2', 'SPM', 'TEMP', 'THC', 'WD', 'WS'] 所沢市東所沢
# ['CH4', 'CO', 'HUM', 'NMHC', 'NO', 'NO2', 'NOX', 'OX', 'SO2', 'SPM', 'TEMP', 'THC', 'WD', 'WS'] 草加市西町



dir_path = 'input_data/learning_data/'
# df  = pd.read_csv(dir_path + 'mergefeatures.csv', index_col=0)
# features_name_dict = df.to_dict()['0']

features_name_dict = {'所沢市東所沢_lag=1':"['CH4','HUM','NMHC','NO','NO2','NOX','OX','PM25','SO2','SPM','TEMP','THC','WD','WS']",
                      '所沢市東所沢_lag=2':"['CH4','HUM','NMHC','NO','NO2','NOX','OX','PM25','SO2','SPM','TEMP','THC','WD','WS']",
                      '所沢市東所沢_lag=3':"['CH4','HUM','NMHC','NO','NO2','NOX','OX','PM25','SO2','SPM','TEMP','THC','WD','WS']",
                      "草加市西町_lag=1":"['CH4','CO','HUM','NMHC','NO','NO2','NOX','OX','SO2','SPM','TEMP','THC','WD','WS']",
                      '草加市西町_lag=2':"['CH4','CO','HUM','NMHC','NO','NO2','NOX','OX','SO2','SPM','TEMP','THC','WD','WS']",
                      '草加市西町_lag=3':"['CH4','CO','HUM','NMHC','NO','NO2','NOX','OX','SO2','SPM','TEMP','THC','WD','WS']",
                      
                      }

for name, features_name in features_name_dict.items():
  state_name = name.split('_')[0]
  lag = name.split('_')[1].split('=')[-1]
  lag = int(lag) - 1
  features = features_name.replace('[', '').replace(']', '').replace("'","").split(',')
  print(len(features))
  save_dir = os.path.join(dir_path, ','.join(features))
  os.makedirs(save_dir + f'{name}', exist_ok = True)
  save_dir = save_dir + f'_{state_name}/'

  max_training_num = 10
  df_li = []
  new_features = []
  for i in features:
      for u in range(1,25):
          new_feature = i + '_' + str(u).zfill(2)
          new_features.append(new_feature)

  ra = train(dir_path + f'{state_name}_learning_24.csv',  dir_path + f'{state_name}_predict_24.csv', features = new_features, lag = lag)

  for training_num in range(max_training_num):
      ra.start_train()
      out_df = ra.start_predict()
      df_li.append(out_df)

  out_df = pd.concat(df_li)
  out_df = out_df.mean()
  out_df['高濃度追跡率'] = (out_df['高濃度追跡'] / out_df['高濃度出現回数']) * 100
  out_df.to_csv(save_dir + 'high_concent_check.csv')

  shap_high_df = pd.read_csv(save_dir + 'shap_high_explanation.csv', index_col = 0, header = 0)
  shap_df = pd.read_csv(save_dir + 'shap_explanation.csv', index_col = 0, header = 0)

  shap_high_df.abs().mean().to_csv(save_dir + 'shap_mean_high_explanation.csv')
  pd.DataFrame(shap_high_df.abs().mean()).sort_values(by = 0, ascending = False).head(20).to_csv(save_dir + 'shap_20_mean_high_explanation.csv')
  shap_df.abs().mean().to_csv(save_dir + 'shap_mean_explanation.csv')
  pd.DataFrame(shap_df.abs().mean()).sort_values(by = 0, ascending = False).head(20).to_csv(save_dir + 'shap_20_mean_explanation.csv')