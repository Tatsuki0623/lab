from datetime import date
import pandas as pd
from datetime import timedelta
from glob import glob
from sklearn.metrics import root_mean_squared_error

check_dict_re = {}
check_dict_pr = {}
check_df_pr_li = []
check_df_re_li = []
out_df_li = []

path_li = glob("out_data/results/DNN/god/*.csv")
for path in path_li:
    df = pd.read_csv(path, index_col = 0, header = 0)
    rmse = root_mean_squared_error(df["obs"], df["predict"])
    idx_num = int((len(df.index.to_list()) / 24) + 1)
    name = path.split("\\")[-1].split(".")[0]

    target_day = date(2019, 4, 1)
    days = timedelta(days = 1)

    all_recall_num = 0
    recall_num = 0
    all_precision_num = 0
    precision_num = 0
    for i in range(idx_num):
        target_str = str(target_day)
        target_df = df[df.index.str.contains(target_str)]
        target_precision = target_df.query('predict >= 100')
        target_recall = target_df.query('obs >= 100')

        if not target_precision.empty:
            all_precision_num += 1
            precision_obs = target_precision.query('obs >= 100')
            if not precision_obs.empty:
                precision_num += 1
                check_dict_pr[target_day] = "○"
            else:
                check_dict_pr[target_day] = "×"
        else:
            check_dict_pr[target_day] = "-"
        
        if not target_recall.empty:
            all_recall_num += 1
            recall_predict = target_recall.query('predict >= 100')
            if not recall_predict.empty:
                recall_num += 1
                check_dict_re[target_day] = "○"
            else:
                check_dict_re[target_day] = "×"
        else:
            check_dict_re[target_day] = "-"
        
        target_day += days
    
    check_df_re = pd.DataFrame(check_dict_re, index = [name]).T
    check_df_pr = pd.DataFrame(check_dict_pr, index = [name]).T

    check_df_pr_li.append(check_df_pr)
    check_df_re_li.append(check_df_re)

    if all_recall_num == 0:
        recall = 0
    else:
        recall = (recall_num / all_recall_num) * 100
    
    if all_precision_num == 0:
        precision = 0
    else:
        precision = (precision_num / all_precision_num) * 100
    
    if (recall + precision) == 0:
        compa = 0
    else:
        compa = (2 * recall * precision) / (recall + precision)

    high_concent_dict = {
                        'RMSE': rmse,
                        '再現率': recall,
                        '適合率': precision,
                        '調和平均': compa
                        }

    out_df = pd.DataFrame(high_concent_dict, [name])
    out_df_li.append(out_df)

out = pd.concat(out_df_li, axis = 0)
out.to_csv("out_data/results/DNN/god/モデル評価.csv")

out_check_df_pr = pd.concat(check_df_pr_li, axis = 1)
out_check_df_pr.to_csv("out_data/results/DNN/god/適合率（日別）.csv")
out_check_df_re = pd.concat(check_df_re_li, axis = 1)
out_check_df_re.to_csv("out_data/results/DNN/god/再現率（日別）.csv")
