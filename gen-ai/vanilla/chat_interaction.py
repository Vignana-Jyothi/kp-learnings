from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)



def generate_text(prompt, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)


    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def chatbot():
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_text(user_input)
        print("Bot:", response)

chatbot()



