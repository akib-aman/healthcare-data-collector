from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json

# Load the trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./trained_model")
model = GPT2LMHeadModel.from_pretrained("./trained_model")

# Load the JSON form template
json_path = "form-data/form.json"
with open(json_path, "r") as file:
    form_data = json.load(file)

# Function to process AI output and update the form
def update_form(form_data, extracted_fields):
    for key, value in extracted_fields.items():
        # Update characteristics
        for characteristic in form_data["Characteristics"]:
            if characteristic["Name"].lower() == key.lower():
                characteristic["Value"] = value
                break
        else:
            # Update nested form fields (e.g., Age or Ethnicity)
            if key in form_data["Form"]:
                form_data["Form"][key]["SelectedValue"] = value
    return form_data

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
        max_length=50,  # Allow more detailed output
        num_beams=5,  # Use beam search for better accuracy
        no_repeat_ngram_size=2,  # Prevent repetition
        early_stopping=True
    )

    # Decode the generated text and process the output

    # print("\nALL OUTPUTS:")
    # print(outputs[0])

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated AI Output:")
    print(generated_text)

    try:
        # Parse the AI output as JSON (assumes fine-tuned model generates JSON-like output)
        extracted_fields = json.loads(generated_text)
        print("\nExtracted Fields:")
        print(json.dumps(extracted_fields, indent=4))

        # Update the form with the extracted fields
        form_data = update_form(form_data, extracted_fields)

        # Save the updated form
        with open(json_path, "w") as file:
            json.dump(form_data, file, indent=4)
        print("\nForm updated successfully!")

    except json.JSONDecodeError:
        print("Error: AI output could not be parsed into valid JSON. Please check the model's output.")

# End of script
