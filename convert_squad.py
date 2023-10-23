import json
import csv

# Open and read the JSON file
with open('dataset/train-v2.0_SQuAD.json', 'r') as f:
    data = json.load(f)

# Create or overwrite the CSV file
with open('dataset/squad_train_output.csv', 'w', newline='') as csvfile:
    fieldnames = ['question', 'answer']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # writes the headers

    # Traverse the JSON structure to extract desired data
    for entry in data['data']:
        for paragraph in entry['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                # Assuming that there's always at least one answer
                if len(qa['answers']) > 0:
                    answer = qa['answers'][0]['text']
                    writer.writerow({'question': question, 'answer': answer})
