import pandas as pd

features = []
features_name = ["OX", "NOX", "NMHC", "TEMP"]
for i in features_name:
    for u in range(1,25):
        new_features = i + '_' + str(u).zfill(2)
        features.append(new_features)

df = pd.DataFrame(features).T
df.to_csv("out_data/test_data/brank.csv", encoding = "sjis")
