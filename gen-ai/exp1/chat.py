from fastapi import FastAPI, Request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging
import time

app = FastAPI()

model_path = "./gpt2-finetuned"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
logging.basicConfig(level=logging.INFO)


@app.post("/chat")
async def chat(request: Request):
    start_time = time.time()
    data = await request.json()
    question = data.get("question")
    inputs = tokenizer.encode(question, return_tensors="pt")
    
    outputs = model.generate(inputs, max_length=500, num_return_sequences=1, temperature=0.7, top_k=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    end_time = time.time()
    logging.info(f"Response time: {end_time - start_time:.2f} seconds")
    
    return {"My Answer": response}
