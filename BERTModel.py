import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import numpy as np

# 1. read data
train_df = pd.read_csv("dataset/data_for_model/cleaned_train.csv")
test_df = pd.read_csv("dataset/data_for_model/cleaned_test.csv")

# 2. pre process the data
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length = 100   # max length for sentence


def encode_data(df):
    input_ids = []
    attention_masks = []
    labels = df["is_human"].values

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
    labels = torch.tensor(labels)

    return TensorDataset(input_ids, attention_masks, labels)


train_dataset = encode_data(train_df)
test_dataset = encode_data(test_df)

batch_size = 32

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# 3. create model bert
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    output_attentions=False,
    output_hidden_states=False
)
# model.to('cuda')
model.to('cpu')

optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# epochs = 3
epochs = 1
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 4. train the model
model.train()

for epoch in range(epochs):
    for batch in train_dataloader:
        batch = tuple(t.to('cpu') for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        model.zero_grad()



# 5. evaluate the model and save predictions
model.eval()

predictions = []
true_labels = []
questions = []
answers = []

for batch in test_dataloader:
    batch = tuple(t.to('cpu') for t in batch)
    b_input_ids, b_input_mask, b_labels = batch

    # Decode the input ids back to text and collect questions and answers
    texts = [tokenizer.decode(input_id, skip_special_tokens=True) for input_id in b_input_ids]
    questions.extend(test_df["question"].iloc[len(questions):len(questions) + len(texts)].tolist())
    answers.extend(texts)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    pred_labels = np.argmax(logits, axis=1).flatten().tolist()
    predictions.extend(pred_labels)
    true_labels.extend(label_ids.flatten().tolist())

# Create a DataFrame with the questions, answers, and the predicted is_human labels
results_df = pd.DataFrame({
    'question': questions,
    'answer': answers,
    'predicted_is_human': predictions  # Directly use the predictions list
})

# Save the results to a new CSV file
results_df.to_csv("dataset/data_for_model/test_result_predictions.csv", index=False)

# print out result
test_accuracy = accuracy_score(true_labels, predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")


# save the model
# Save only the model parameters (recommended)
torch.save(model.state_dict(), 'model/bert_model_state_dict.pth')











# # 5. evaluate the model
# model.eval()
#
# predictions = []
# true_labels = []
#
# for batch in test_dataloader:
#     batch = tuple(t.to('cpu') for t in batch)
#     b_input_ids, b_input_mask, b_labels = batch
#
#     with torch.no_grad():
#         outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
#     logits = outputs[0]
#     logits = logits.detach().cpu().numpy()
#     label_ids = b_labels.to('cpu').numpy()
#
#     predictions.extend(np.argmax(logits, axis=1).flatten().tolist())
#     true_labels.extend(label_ids.flatten().tolist())
#
# test_accuracy = accuracy_score(true_labels, predictions)
# print(f"Test Accuracy: {test_accuracy:.2f}")
