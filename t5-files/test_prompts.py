from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

# Load the trained model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("./t5-trained-model")
model = T5ForConditionalGeneration.from_pretrained("./t5-trained-model")

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

    # Prepare the input for T5
    input_text = f"Input: {prompt}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Generate a response
    outputs = model.generate(
        inputs["input_ids"],
        max_length=50,  # Allow more detailed output
        num_beams=5,  # Use beam search for better accuracy
        no_repeat_ngram_size=2,  # Prevent repetition
        early_stopping=True
    )

    # Decode the generated text and process the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated AI Output:")
    print(generated_text)

    try:
        # Ensure JSON output format by adding braces if needed
        if not generated_text.startswith("{"):
            generated_text = "{" + generated_text + "}"
        extracted_fields = json.loads(generated_text)
        print("\nExtracted Fields:")
        print(json.dumps(extracted_fields, indent=4))

        # Update the form with the extracted fields
        form_data = update_form(form_data, extracted_fields)

        # Save the updated form
        with open(json_path, "w") as file:
            json.dump(form_data, file, indent=4)
        print("\nForm updated successfully!")

    except json.JSONDecodeError as e:
        print(f"Error: AI output could not be parsed into valid JSON. Details: {e}")


# End of script
