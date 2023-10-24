import os
import json
import csv


source_directory = "dataset/exist_stackexchange"
destination_directory = 'dataset/transfer_csv_stackexchange'


for filename in os.listdir(source_directory):
    if filename.endswith(".json"):
        filepath = os.path.join(source_directory, filename)

        # read json
        with open(filepath, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)

            csv_filename = os.path.splitext(filename)[0] + ".csv"
            csv_filepath = os.path.join(destination_directory, csv_filename)

            with open(csv_filepath, 'w', newline='', encoding='utf-8') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["question", "answer"])

                for key, value in data.items():
                    writer.writerow([key, value])

    print(f"Finished transfer {filename}")


print("\n========Finished writing csv from json========")
