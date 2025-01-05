import json
import torch
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel,
    T5Tokenizer, T5ForConditionalGeneration
)

############################################
# 1. LOAD MODELS & TOKENIZERS
############################################

# GPT model (trained on Q&A or “wordy” responses)
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./gpt-trained-model") 
gpt_model = GPT2LMHeadModel.from_pretrained("./gpt-trained-model")

# T5 model (trained on extracting form fields)
t5_tokenizer = T5Tokenizer.from_pretrained("./t5-trained-model")
t5_model = T5ForConditionalGeneration.from_pretrained("./t5-trained-model")

# Load your JSON form template
json_path = "form-data/form.json"
with open(json_path, "r") as file:
    form_data = json.load(file)

############################################
# 2. HELPER FUNCTIONS
############################################

def is_question(prompt: str) -> bool:
    """
    Naive classifier: checks if user input is likely a question.
    Customize this logic or replace with a proper classifier if needed.
    """
    prompt_lower = prompt.strip().lower()
    if prompt_lower.endswith('?'):
        return True
    # Check for typical question words:
    question_words = ["why", "how", "what", "who", "where", "when"]
    if any(qw in prompt_lower for qw in question_words):
        return True
    return False

def update_form(form_data, extracted_fields):
    """
    Update form_data with the fields extracted by T5.
    """
    for key, value in extracted_fields.items():
        # Update 'Characteristics'
        for characteristic in form_data["Characteristics"]:
            if characteristic["Name"].lower() == key.lower():
                characteristic["Value"] = value
                break
        else:
            # Update nested 'Form' fields (e.g., Age or Ethnicity)
            if key in form_data["Form"]:
                form_data["Form"][key]["SelectedValue"] = value
    return form_data

############################################
# 3. INFERENCE FUNCTIONS
############################################

def generate_with_gpt(prompt: str) -> str:
    """
    Generates a response using GPT for Q&A or wordy answers.
    """
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
    # (Potentially move model to GPU if available)
    outputs = gpt_model.generate(
        inputs["input_ids"],
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_with_t5(prompt: str) -> str:
    """
    Generates an extraction or short result using T5.
    Example input: "my name is John" -> T5 outputs JSON: {"Name": "John"}
    """
    # For T5, we often prepend a task prefix, e.g., 'extract: <prompt>'
    # if your fine-tuning approach used a prefix. Adjust if needed.
    # If not, you can pass prompt directly.
    input_text = f"extract: {prompt}"
    inputs = t5_tokenizer.encode(prompt, return_tensors="pt")
    outputs = t5_model.generate(
        inputs,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

############################################
# 4. MAIN INTERACTION LOOP
############################################
print("Enter your prompt (type 'exit' to quit):")

while True:
    prompt = input("\nYour prompt: ")
    if prompt.lower() == 'exit':
        print("Exiting the prompt console.")
        break

    # 4a. Determine if it's a question or statement
    if is_question(prompt):
        # Use GPT
        generated_text = generate_with_gpt(prompt)
        print("\n[GPT ANSWER]:")
        print(generated_text)

    else:
        # Use T5
        generated_text = generate_with_t5(prompt)
        print("\n[T5 OUTPUT]:")
        print(generated_text)

        # 4b. Try to parse T5 output as JSON and update the form if valid
        try:
            # Ensure valid JSON format by wrapping in curly braces if needed
            if ":" in generated_text and not generated_text.startswith("{"):
                generated_text = "{" + generated_text.strip() + "}"

            extracted_fields = json.loads(generated_text)
            print("\nExtracted Fields:")
            print(json.dumps(extracted_fields, indent=4))

            form_data = update_form(form_data, extracted_fields)

            # Save the updated form
            with open(json_path, "w") as file:
                json.dump(form_data, file, indent=4)

            print("\nForm updated successfully!")

        except json.JSONDecodeError:
            print("Error: T5 output could not be parsed into valid JSON.")
            print("Please check if the model is generating valid JSON.")
