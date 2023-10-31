import pandas as pd


train_df = pd.read_csv('dataset/data_for_model/train.csv')
test_df = pd.read_csv('dataset/data_for_model/test.csv')

# clean: - remove nan
train_df.dropna(subset=['answer'], inplace=True)
test_df.dropna(subset=['answer'], inplace=True)


# clean: - drop duplicates
train_df.drop_duplicates(inplace=True)
test_df.drop_duplicates(inplace=True)


train_df.to_csv('dataset/data_for_model/cleaned_train.csv', index=False)
test_df.to_csv('dataset/data_for_model/cleaned_test.csv', index=False)
