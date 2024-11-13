import pandas as pd
import os

dir_path = 'out_data/results/DNN/'

dir_names = [f for f in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, f))]
state_dict = {}

for dir_name in dir_names:
    state_name_li = dir_name.split('_')
    state_name = state_name_li[-2] +  '_' + state_name_li[-1]
    target_file_path = 'out_data/results/DNN/' + dir_name + '/shap_20_mean_explanation.csv'
    index_df = pd.read_csv(target_file_path, index_col = 0, header = 0)
    index = index_df.index.to_list()
    index = [i.replace('SHAP_', '') for i in index]
    state_dict[state_name] = str(index)


state_df = pd.DataFrame(state_dict, index = [0]).T
print(state_df)
state_df.to_csv('out_data/results/DNN/AllFeatures_low.csv')


    