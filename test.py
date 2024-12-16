from datetime import date
import pandas as pd
from datetime import timedelta
from sklearn.metrics import root_mean_squared_error

target_day = date(2019, 4, 1)
days = timedelta(days = 1)

df = pd.read_csv("out_data/results/DNN/上位20個/幸手_lag=1/out.csv", index_col = 0, header = 0)

all_recall_num = 0
recall_num = 0
all_precision_num = 0
precision_num = 0

for i in range(365):
    target_str = str(target_day)
    target_df = df[df.index.str.contains(target_str)]
    target_precision = target_df.query('predict >= 100')
    target_recall = target_df.query('obs >= 100')

    if not target_precision.empty:
        all_precision_num += 1
        precision_obs = target_precision.query('obs >= 100')
        if not precision_obs.empty:
            precision_num += 1
    
    if not target_recall.empty:
        all_recall_num += 1
        recall_predict = target_recall.query('predict >= 100')
        if not recall_predict.empty:
            recall_num += 1
    
    target_day += days

recall = (recall_num / all_recall_num) * 100
precision = (precision_num / all_precision_num) * 100

compa = (2 * recall * precision) / (recall + precision)

high_concent_dict = {
                    '再現率': recall,
                    '適合率': precision,
                    '調和平均': compa
                    }

out = pd.DataFrame(high_concent_dict, [0])
print(out)