from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 model and tokenizer
model_name = 'gpt2'  # or 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

def generate_text(prompt, max_length=50):
    # Encode the input prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = tokenizer.encode(prompt, return_tensors='pt')

    # Generate the output text
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        pad_token_id=tokenizer.eos_token_id,
        attention_mask=attention_mask
    )

    # Decode the generated text
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Example usage
prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)
