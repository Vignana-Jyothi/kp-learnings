import os
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

# Define the paths to the dataset files
data_dir = os.path.expanduser('~/kp-learnings/kaggle/sf-qa-dataset')
train_file = os.path.join(data_dir, 'train-v1.1.json')
validation_file = os.path.join(data_dir, 'dev-v1.1.json')

# Load the dataset
dataset = load_dataset('json', data_files={'train': train_file, 'validation': validation_file})

# Inspect the structure of one example
print(dataset['train'][0])

# Tokenization
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def preprocess_function(examples):
    contexts = []
    questions = []
    for example in examples['paragraphs']:
        context = example['context']
        qas = example['qas']
        for qa in qas:
            question = qa['question']
            contexts.append(context)
            questions.append(question)
    
    # Combine context and question for tokenization
    inputs = [q + " " + c for q, c in zip(questions, contexts)]
    model_inputs = tokenizer(inputs, truncation=True, padding="max_length", max_length=128)
    return model_inputs

# Flatten the dataset structure
def flatten_dataset(batch):
    flattened_data = []
    for example in batch['data']:
        for paragraph in example['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                flattened_data.append({'context': context, 'question': qa['question']})
    return flattened_data

# Apply the flatten function to the dataset
flattened_train = dataset['train'].map(flatten_dataset, batched=True)
flattened_validation = dataset['validation'].map(flatten_dataset, batched=True)

# Apply the preprocess function to the flattened dataset
tokenized_train = flattened_train.map(preprocess_function, batched=True, remove_columns=["context", "question"])
tokenized_validation = flattened_validation.map(preprocess_function, batched=True, remove_columns=["context", "question"])

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
