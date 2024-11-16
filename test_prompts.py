from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Console interaction loop
print("Enter your prompt (type 'exit' to quit):")
while True:
    # Get user input
    prompt = input("\nYour prompt: ")
    if prompt.lower() == 'exit':
        print("Exiting the prompt console.")
        break
    
    # Tokenize the prompt and generate response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=50, num_return_sequences=1)
    
    # Decode and print the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated response:")
    print(generated_text)
