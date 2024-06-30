import pandas as pd
import torch
from transformers import TapasTokenizer, TapasForQuestionAnswering

# Load the model and tokenizer
model_name = "google/tapas-base-finetuned-wtq"
tokenizer = TapasTokenizer.from_pretrained(model_name)
model = TapasForQuestionAnswering.from_pretrained(model_name)

# Load your table from a CSV file
table = pd.read_csv('simple-table-data.csv')

# Convert all cells in the table to strings
table = table.astype(str)


# Adjust the queries to match the CSV data
queries = [
    "Which city does Bob belong to?",
    "What is the age of Alice?",
    "Where does David stay?",
    "Who stays in Chicago?",
    "Who is older, Bob or Eve?"
]

# Process each query separately to avoid issues with input size
for query in queries:
    inputs = tokenizer(table=table, queries=query, padding="max_length", truncation=True, return_tensors="pt")

    # Perform the forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answers
    predicted_answer_coordinates, predicted_aggregation_indices = tokenizer.convert_logits_to_predictions(
        inputs,
        outputs.logits.detach(),
        outputs.logits_aggregation.detach()
    )

    # Find the predicted answer text
    answers = []
    for coordinates in predicted_answer_coordinates:
        if len(coordinates) == 1:
            # only a single cell:
            answer = table.iat[coordinates[0]]
        else:
            # multiple cells
            answer = " ".join([table.iat[coordinate] for coordinate in coordinates])
        answers.append(answer)

    # Print the answer for the current query
    print(f"Question: {query}")
    for answer in answers:
        print(f"Answer: {answer}")
