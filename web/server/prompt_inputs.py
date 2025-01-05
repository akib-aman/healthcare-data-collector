import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration

# Load Models & Tokenizers
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./gpt-trained-model")
gpt_model = GPT2LMHeadModel.from_pretrained("./gpt-trained-model")
t5_tokenizer = T5Tokenizer.from_pretrained("./t5-trained-model")
t5_model = T5ForConditionalGeneration.from_pretrained("./t5-trained-model")

# Load JSON form template
json_path = "form-data/form.json"
with open(json_path, "r") as file:
    form_data = json.load(file)

def is_question(prompt: str) -> bool:
    """
    Checks if user input is likely a question.
    """
    prompt_lower = prompt.strip().lower()
    if prompt_lower.endswith('?'):
        return True
    question_words = ["why", "how", "what", "who", "where", "when"]
    return any(qw in prompt_lower for qw in question_words)

def generate_with_gpt(prompt: str) -> str:
    """
    Generates a response using GPT for Q&A or wordy answers.
    """
    inputs = gpt_tokenizer(prompt, return_tensors="pt")
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
    """
    input_text = f"extract: {prompt}"
    inputs = t5_tokenizer.encode(input_text, return_tensors="pt")
    outputs = t5_model.generate(
        inputs,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("OUTPUT:" + t5_output)
    # Ensure the output is valid JSON
    if ":" in t5_output and not t5_output.startswith("{"):
        t5_output = "{" + t5_output.strip() + "}"
    
    return t5_output

def handle_prompt(prompt: str) -> str:
    """
    Routes the prompt to the appropriate model and returns the response.
    """
    if is_question(prompt):
        return generate_with_gpt(prompt)
    else:
        t5_output = generate_with_t5(prompt)
        try:
            extracted_fields = json.loads(t5_output)
            update_form(form_data, extracted_fields)
            
            # Save the updated form
            with open(json_path, "w") as file:
                json.dump(form_data, file, indent=4)

            return json.dumps(extracted_fields, indent=4)
        except json.JSONDecodeError as e:
            print("JSON Decode Error:", e)
            return f"Error: T5 output could not be parsed into valid JSON. Output was: {t5_output}"

def update_form(form_data, extracted_fields):
    """
    Update form_data with the fields extracted by T5.
    """
    print("FIELDS:" + str(extracted_fields))
    for key, value in extracted_fields.items():
        for characteristic in form_data["Characteristics"]:
            if characteristic["Name"].lower() == key.lower():
                characteristic["Value"] = value
                break
        else:
            if key in form_data["Form"]:
                form_data["Form"][key]["SelectedValue"] = value
    return form_data
