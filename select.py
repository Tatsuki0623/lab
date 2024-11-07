import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('python-code/lab/out_data/individual_value_harumi/material_concent.csv', header = 0, index_col = 0)

a = df['OX'][10:]
for i in df.columns:
    print(i)
    b = df[i][:-10]
    plt.figure()
    plt.scatter(b, a)
    plt.xlabel(i)
    plt.ylabel('OX')
    plt.savefig('python-code/lab/out_data/相関図/晴海/10時間前/OX vs ' + i + '.png')