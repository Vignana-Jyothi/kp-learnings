from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


app = FastAPI()

model_path = "./gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"answer": response}

# Run the app
# uvicorn script_name:app --reload
