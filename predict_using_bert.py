import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score
import numpy as np


# Define the path to your saved model and test dataset
MODEL_PATH = 'model/bert_model_state_dict.pth'
TEST_DATA_PATH = 'dataset/data_for_model/cleaned_test.csv'
RESULTS_PATH = 'dataset/data_for_model/test_result_predictions_new_2.csv'

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)

# Load the state dict into the model
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()  # Set model to evaluation mode

# Load the test data
test_df = pd.read_csv(TEST_DATA_PATH)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 100

# Function to encode the data
def encode_data(df):
    input_ids = []
    attention_masks = []
    for answer in df["answer"]:
        encoded_dict = tokenizer.encode_plus(
            answer,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['is_human'].values)
    return TensorDataset(input_ids, attention_masks, labels)

# Prepare the test data loader
test_dataset = encode_data(test_df)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Prepare to collect predictions and actual labels
predictions = []
true_labels = []
questions = []
answers = []

# Evaluate the model
for batch in test_dataloader:
    b_input_ids, b_attention_mask, b_labels = [t.to('cpu') for t in batch]

    # Decode the input ids back to strings
    questions_batch = test_df["question"].iloc[:len(b_input_ids)].tolist()
    answers_batch = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in b_input_ids]
    questions.extend(questions_batch)
    answers.extend(answers_batch)
    test_df = test_df.iloc[len(b_input_ids):]

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)

    logits = outputs[0]
    predictions.extend(logits.argmax(axis=-1).flatten().tolist())
    true_labels.extend(b_labels.numpy().flatten().tolist())

# Create a DataFrame with the results
results_df = pd.DataFrame({
    'question': questions,
    'answer': answers,
    'is_human': true_labels,
    'predicted_is_human': predictions
})

# results_df.drop(results_df.columns[-1], axis=1, inplace=True)

# Save the DataFrame to a CSV file
results_df.to_csv(RESULTS_PATH, index=False)

# Calculate and print the accuracy
accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {accuracy:.2f}")
