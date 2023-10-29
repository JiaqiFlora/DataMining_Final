import pandas as pd
import openai
import configparser
import os

config = configparser.ConfigParser()
config.read('config.txt')
api_key = config['openai']['api_key']

openai.api_key = api_key

original_file_path = "dataset/stackexchange/temp.csv"
df = pd.read_csv(original_file_path)

if 'question' not in df.columns:
    print("Error: No 'question' column found in the CSV!")
    exit()

base_name = os.path.basename(original_file_path)
file_prefix, file_extension = os.path.splitext(base_name)
output_file_path = f"dataset/openai_answer/openai_{file_prefix}.csv"


header_written = False
for question in df['question']:
    response = openai.Completion.create(engine='ada',prompt=question, max_tokens=150)
    answer = response.choices[0].text.strip()
    current_df = pd.DataFrame([{"question": question, "answer": answer}])
    print(current_df)

    # Append to the output file
    current_df.to_csv(output_file_path, mode='a', header=not header_written, index=False)
    header_written = True



