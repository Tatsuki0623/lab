import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

name = 'kounosu_corr'
flags = [True, False]

# データの読み込み
df = pd.read_csv(f'python-code/lab/out_data/verification/{name}.csv', index_col=0, parse_dates=True)

max_lag = 24

# 除外する列
exclude_columns = ['CO', 'NETR', 'SUN', 'UV']

l_start_date = 2016
l_last_date = l_start_date + 3

# 除外する列を除いたデータフレームを作成
df = df.drop(columns = exclude_columns)
df = df.loc[str(l_start_date) + '-01-01 00:00:00': str(l_last_date) + '-12-31 23:00:00']
lagged_df = df.copy()

for n in range(1, max_lag + 1):
    lagged_df[f'OX_lag{n}'] = df['OX'].shift(-n)

# タイムラグ列を追加したデータをCSVに出力
output_file = f'python-code/lab/out_data/verification/lagged_data_1_to_24_hours_{name}.csv'
# lagged_df.to_csv(output_file)

for flag in flags:
    if flag:

        # 各変数とOXのタイムラグ列との相関係数を計算
        correlations = {}
        for col in lagged_df.columns:
            if col.startswith('OX_lag'):
                continue
            correlations[col] = []
            for n in range(1, max_lag + 1):
                corr = int(lagged_df[col].corr(lagged_df[f'OX_lag{n}']) * 1000) / 1000
                correlations[col].append(corr)

        # 相関係数をデータフレームに変換
        corr_df = pd.DataFrame(correlations, index=[f'OX_lag{n}' for n in range(1, max_lag + 1)])
        # ヒートマップの作成
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation of Variables with Lagged OX Values {name}')
        plt.xlabel('Variables')
        plt.ylabel('OX Lag')
        plt.savefig(f'python-code/lab/out_data/verification/{name}.png')
    else:
        # 各変数とOXのタイムラグ列との決定係数（R²）を計算
        r2_scores = {}
        for col in lagged_df.columns:
            if col.startswith('OX_lag'):
                continue
            r2_scores[col] = []
            for n in range(1, max_lag + 1):
                # 欠損値を除外
                valid_data = lagged_df[[col, f'OX_lag{n}']].dropna()
                if valid_data.shape[0] > 0:
                    X = valid_data[[col]].values
                    y = valid_data[f'OX_lag{n}'].values
                    model = LinearRegression().fit(X, y)
                    r2_scores[col].append(int(r2_score(y, model.predict(X)) * 1000) / 1000)
                else:
                    r2_scores[col].append(None)

        # 決定係数（R²）をデータフレームに変換
        corr_df = pd.DataFrame(r2_scores, index=[f'OX_lag{n}' for n in range(1, max_lag + 1)])
        
        # ヒートマップの作成
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0)
        plt.title(f'Correlation of Variables with Lagged OX Values {name}')
        plt.xlabel('Variables')
        plt.ylabel('OX Lag')
        plt.savefig(f'python-code/lab/out_data/verification/{name}_R2.png')

