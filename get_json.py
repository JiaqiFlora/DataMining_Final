# import pandas as pd
#
#
# df = pd.read_csv('dataset/data_for_model/test_result_predictions_new.csv')
# df.drop(df.columns[-1], axis=1, inplace=True)
# df_sample = df.sample(n=200)
# df_sample.to_json('dataset/json/data_unity.json', orient='records', lines=True)



import pandas as pd
import json


df = pd.read_csv('dataset/data_for_model/data_for_proj.csv')
# df.drop(df.columns[-1], axis=1, inplace=True)
# num_lines = min(15, len(df))
# df_sample = df.sample(n=num_lines)
json_data = df.to_json(orient='records', lines=False)
with open('dataset/json/data_unity.json', 'w') as json_file:
    json_file.write(json_data)

json_list = json.loads(json_data)
print(json_list)

