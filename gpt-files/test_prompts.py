from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json
import torch

# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./trained_model")
model = GPT2LMHeadModel.from_pretrained("./trained_model")

# Load the JSON form template
json_path = "form-data/form.json"
with open(json_path, "r") as file:
    form_data = json.load(file)

# Function to extract age from a sentence
def extract_age(prompt):
    import re
    match = re.search(r'\b(\d{1,2})\b', prompt)
    if match:
        return match.group(1)
    return None

# Console interaction loop
print("Enter your prompt (type 'exit' to quit):")
while True:
    prompt = input("\nYour prompt: ")
    if prompt.lower() == 'exit':
        print("Exiting the prompt console.")
        break

    # Tokenize the prompt and generate a response
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_length=20,  # Limit the output length
        num_beams=5,  # Use beam search
        no_repeat_ngram_size=2,  # Prevent repetition
        early_stopping=True
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated response:")
    print(generated_text)

    # Extract age from the prompt and update the JSON form
    age = extract_age(generated_text)
    if age:
        form_data["Characteristics"][0]["Value"] = age
        with open(json_path, "w") as file:
            json.dump(form_data, file, indent=4)
        print(f"END -- Age '{age}' written to form.")
    else:
        print("No age found in the prompt.")
