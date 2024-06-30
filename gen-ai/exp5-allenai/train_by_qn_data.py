import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
from datasets import DatasetDict, Dataset

# Load your custom dataset
def load_custom_dataset(file_path):
    with open(file_path, 'r') as f:
        dataset_dict = json.load(f)
    return dataset_dict

# Assuming your dataset file path is 'custom_qa_dataset.json'
custom_dataset = load_custom_dataset('custom_qa_dataset.json')

# Convert to a list of dictionaries with context, question, and answers
def convert_to_squad_format(custom_dataset):
    contexts = []
    questions = []
    answers = []
    
    for data in custom_dataset["data"]:
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                for answer in qa["answers"]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append({
                        "text": answer["text"],
                        "answer_start": answer["answer_start"]
                    })
    
    return {
        "context": contexts,
        "question": questions,
        "answers": answers
    }

squad_format_dataset = convert_to_squad_format(custom_dataset)

# Convert to Hugging Face dataset format
dataset = DatasetDict({"train": Dataset.from_dict(squad_format_dataset)})

# Load the tokenizer and model
model_name = "allenai/longformer-base-4096"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Tokenize the dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=4096,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, answer in enumerate(examples["answers"]):
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is out of the context, label it (0, 0)
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

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["train"],
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./fine-tuned-longformer")
tokenizer.save_pretrained("./fine-tuned-longformer")
