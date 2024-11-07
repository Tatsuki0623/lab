import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt

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

class OzonLSTMPredictor:
    def __init__(self, model_path, time_steps = 24, features = [], scaler_flag = True):
        self.model_path = model_path
        self.time_steps = time_steps
        self.features = features
        self.scaler_flag = scaler_flag
        self.scaler_norm = MinMaxScaler()
        self.scaler_stand = StandardScaler()
        self.model = self.load_model()

    def load_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OzonLSTM(input_size = len(self.features), hidden_size = 64, output_size = 1, num_layers = 3).to(device)

        model.load_state_dict(torch.load(self.model_path))
        model.eval()

        return model

    def create_sequences(self, data_path):
        X_sequences = []
        y_sequences = []

        data = pd.read_csv(data_path, index_col = None, header = 0)
        if self.scaler_flag:
            scaled_data = pd.DataFrame(self.scaler_norm.fit_transform(data))
        else:
            scaled_data = pd.DataFrame(self.scaler_stand.fit_transform(data))

        last_data = len(data) - self.time_steps

        OX = 0

        for i in range(last_data):
            target_point = i + self.time_steps
            X_sequences.append(scaled_data.iloc[i:target_point].values)
            y_sequences.append(scaled_data.iloc[target_point, OX])

        return np.array(X_sequences), np.array(y_sequences)

    def predict(self, data_path):
        criteria = nn.MSELoss()

        X_sequences, y_sequences = self.create_sequences(data_path)
        X_sequences = torch.tensor(X_sequences, dtype = torch.float)
        y_sequences = torch.tensor(y_sequences, dtype = torch.float)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        X_sequences = X_sequences.to(device)
        y_sequences = y_sequences.to(device)

        with torch.no_grad():
            predictions = self.model(X_sequences)
        
        print(f'test case MSE = {criteria(predictions, y_sequences)}')
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_sequences.cpu().numpy(), label = 'True Values')
        plt.plot(predictions.cpu().numpy(), label = 'Predicted Values')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.title('True vs Predicted Values')
        plt.legend()
        plt.show()

        return predictions
    
model_path = 'python-code/lab/out_data/model/ozone_prediction_lstm_model_harumi_rayer_5_HUM_NO_3.pth'
data_path = 'python-code/lab/input_data/learning_data/machine_learning_harumi_predict_2019.csv'
features = ['OX', 'NO', 'TEMP', 'HUM', 'sin_day', 'cos_day', 'sin_hour', 'cos_hour']
time_steps = 15
scaler_flag = True

# 予測
predictor = OzonLSTMPredictor(model_path, time_steps, features, scaler_flag)
predictions = predictor.predict(data_path)