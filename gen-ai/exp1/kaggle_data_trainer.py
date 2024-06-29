import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import json
import pandas as pd
import numpy as np

# Define the paths to the dataset files
data_dir = os.path.expanduser('~/kp-learnings/kaggle/sf-qa-dataset')
train_file = os.path.join(data_dir, 'train-v1.1.json')
validation_file = os.path.join(data_dir, 'dev-v1.1.json')

# Function to convert SQuAD JSON to DataFrame for training
def squad_json_to_dataframe_train(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'], verbose=1):
    if verbose:
        print("Reading the json file")
    with open(input_file_path, 'r') as f:
        file = json.load(f)
    if verbose:
        print("processing...")
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])
    idx = np.repeat(r['context'].values, r.qas.str.len())
    ndx = np.repeat(m['id'].values, m['answers'].str.len())
    m['context'] = idx
    js['q_idx'] = ndx
    main = pd.concat([m[['id', 'question', 'context']].set_index('id'), js.set_index('q_idx')], axis=1, sort=False).reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

# Function to convert SQuAD JSON to DataFrame for validation
def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'], verbose=1):
    if verbose:
        print("Reading the json file")
    with open(input_file_path, 'r') as f:
        file = json.load(f)
    if verbose:
        print("processing...")
    js = pd.json_normalize(file, record_path)
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    main['c_id'] = main['context'].factorize()[0]
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

# Convert JSON to DataFrame
train_df = squad_json_to_dataframe_train(input_file_path=train_file)
dev_df = squad_json_to_dataframe_dev(input_file_path=validation_file)

# Flatten the dataset structure
def flatten_dataset(df):
    flattened_data = []
    for _, row in df.iterrows():
        flattened_data.append({'context': row['context'], 'question': row['question']})
    return flattened_data

flattened_train = flatten_dataset(train_df)
flattened_validation = flatten_dataset(dev_df)

# Extract the combined context and question strings for tokenization
def combine_context_and_question(data):
    return [f"{item['question']} {item['context']}" for item in data]

combined_train = combine_context_and_question(flattened_train)
combined_validation = combine_context_and_question(flattened_validation)

# Load the dataset
dataset = load_dataset('json', data_files={'train': train_file, 'validation': validation_file})

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_function(examples):
    inputs = examples['text']
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    return model_inputs

# Convert the combined text to a format suitable for the tokenizer
train_data = {'text': combined_train}
validation_data = {'text': combined_validation}

# Apply the preprocess function to the combined text
tokenized_train = preprocess_function(train_data)
tokenized_validation = preprocess_function(validation_data)

# Fine-Tuning
model = GPT2LMHeadModel.from_pretrained('gpt2')
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
)

trainer.train()

# Save Model
model.save_pretrained('./gpt2-finetuned')
tokenizer.save_pretrained('./gpt2-finetuned')

# Testing
question = "What are the different types of generative AI models?"
inputs = tokenizer.encode(question, return_tensors="pt")
outputs = model.generate(inputs, max_length=150, temperature=0.7, top_k=50, top_p=0.9)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
