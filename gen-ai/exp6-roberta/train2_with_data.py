import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict

# Load your custom dataset
data_files = {"data": "custom_qa_dataset.json"}
dataset = load_dataset('json', data_files=data_files)['data']

# Check the number of samples in the dataset
print(f"Number of samples in the dataset: {len(dataset)}")

# Split the dataset into train and validation sets
split_dataset = dataset.train_test_split(test_size=0.2)  # 20% for validation

# Load the tokenizer and model
model_name = 'deepset/roberta-base-squad2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Preprocessing function
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, answer in enumerate(answers):
        start_char = answer[0]["answer_start"]
        end_char = start_char + len(answer[0]["text"])

        sequence_ids = inputs.sequence_ids(i)

        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        if not (offset_mapping[i][context_start][0] <= start_char and offset_mapping[i][context_end][1] >= end_char):
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_idx = context_start
            while offset_mapping[i][start_idx][0] <= start_char:
                start_idx += 1
            start_positions.append(start_idx - 1)

            end_idx = context_end
            while offset_mapping[i][end_idx][1] >= end_char:
                end_idx -= 1
            end_positions.append(end_idx + 1)

    inputs.update({
        "start_positions": start_positions,
        "end_positions": end_positions
    })
    return inputs

# Apply the preprocessing function
tokenized_datasets = split_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],  # Use "test" set for evaluation
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-roberta")
tokenizer.save_pretrained("./fine-tuned-roberta")
