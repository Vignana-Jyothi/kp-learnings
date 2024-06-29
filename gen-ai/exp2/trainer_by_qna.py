from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

# Load dataset
dataset = load_dataset('json', data_files='qa_dataset.json')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large")

# Tokenize the dataset
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        examples["context"],
        questions,
        max_length=512,
        truncation="only_second",
        padding="max_length",
        return_tensors='pt'
    )
    inputs["start_positions"] = [a["answer_start"][0] for a in examples["answers"]]
    inputs["end_positions"] = [a["answer_start"][0] + len(a["text"][0]) for a in examples["answers"]]
    return inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)



# Load the model
model = AutoModelForQuestionAnswering.from_pretrained("microsoft/deberta-large")

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['train'],  # Ideally, you should have a separate validation set
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
