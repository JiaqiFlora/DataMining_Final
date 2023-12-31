import os
import pandas as pd
from sklearn.model_selection import train_test_split


def process_files(folder, label_value):
    all_train_frames = []
    all_test_frames = []

    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)

        # add label
        df = pd.read_csv(filepath)
        df['is_human'] = label_value

        # split train, and test dataset
        train, test = train_test_split(df, test_size=0.2, random_state=42)

        # save split data into each list
        all_train_frames.append(train)
        all_test_frames.append(test)

    return all_train_frames, all_test_frames


# process every file and to get split data
human_train, human_test = process_files("dataset/stackexchange_all", 1)
ai_train, ai_test = process_files("dataset/openai_answer", 0)

# merge human and AI data
merged_train = pd.concat(human_train + ai_train)
merged_test = pd.concat(human_test + ai_test)

# shuffle dataset
merged_train_shuffled = merged_train.sample(frac=1).reset_index(drop=True)
merged_test_shuffled = merged_test.sample(frac=1).reset_index(drop=True)

# save data into files
merged_train_shuffled.to_csv('dataset/data_for_model/train.csv', index=False)
merged_test_shuffled.to_csv('dataset/data_for_model/test.csv', index=False)
