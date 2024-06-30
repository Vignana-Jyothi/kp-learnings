import pandas as pd
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Step 1: Load the model and tokenizer
model_name = "google/tapas-base-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Step 2: Read the table data from a CSV file and convert all cells to strings
table = pd.read_csv('vnr-data.csv').astype(str)

# Step 3: Define the questions
queries = ["What is Designation of C D Naidu", "Which Department Gouthami Koduri belong to?"]

# Step 4: Tokenize the input
inputs = tokenizer(table=table, queries=queries, padding="max_length", return_tensors="pt")

# Step 5: Get predictions
outputs = model(**inputs)
logits = outputs.logits

# Step 6: Post-process the predictions
predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
    inputs,
    outputs.logits.detach(),
    outputs.logits_aggregation.detach()
)

# Extract and print the answers
answers = []
for coordinates in predicted_answer_coordinates:
    if len(coordinates) == 1:
        answer = table.iat[coordinates[0]]
    else:
        answer = " & ".join(table.iat[coord] for coord in coordinates)
    answers.append(answer)

for query, answer in zip(queries, answers):
    print(f"Question: {query}")
    print(f"Answer: {answer}")

# Optional: Save results to a CSV file
df = pd.DataFrame({"Question": queries, "Answer": answers})
df.to_csv("qa_results.csv", index=False)
