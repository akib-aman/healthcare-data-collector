import os
import json
import uuid
import re
from langchain.prompts import PromptTemplate
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration

# ---------------------
# Decision Prompt
# ---------------------
decision_prompt = PromptTemplate(
    input_variables=["user_input"],
    template=(
        "Does this input require data extraction or a detailed response?\n"
        "Input: {user_input}\n"
        "Answer with either 'Data Extraction' or 'Detailed Response'."
    )
)

# ---------------------
# Directory to store session-specific forms
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # The directory of the current script
FORM_DATA_DIR = os.path.join(BASE_DIR, "form-data")    
SESSION_FORMS_DIR = os.path.join(FORM_DATA_DIR, "session-forms")

# ---------------------
# Load Models & Tokenizers
# ---------------------
gpt_tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))
gpt_model = GPT2LMHeadModel.from_pretrained(os.path.join(BASE_DIR, "gpt-trained-model"))

t5_tokenizer = T5Tokenizer.from_pretrained(os.path.join(BASE_DIR, "t5-trained-model"))
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join(BASE_DIR, "t5-trained-model"))

field_commands = {
    "title": "Extract the title from this text: ",
    "firstname": "Extract the firstname from this text: ",
    "lastname": "Extract the lastname from this text: ",
    "age": "Extract the age from this text: ",
    "sex": "Extract the sex from this text: ",
    "genderreassignment": "Extract the gender reassignment from this text: ",
    "marriage/civilpartnership": "Extract the marriage/civil partnership status from this text: ",
    "sexualorientation": "Extract the sexual orientation from this text: ",
    "disability": "Extract the disability from this text: ",
    "religion/belief": "Extract the religion/belief from this text: ",
    "ethnicity": "Extract the ethnicity from this text: ",
    "race": "Extract the race from this text: ",
    "pregnancy/maternity": "Extract the pregnancy/maternity from this text: ",
    "specialrequirements": "Extract the special requirements from this text: ",
}

# Ensure session-forms directory exists
os.makedirs(SESSION_FORMS_DIR, exist_ok=True)

# ---------------------
# Create a Session-Specific JSON
# ---------------------
def create_session_form(form_type: str) -> str:
    """
    Creates a new session-specific JSON file based on data-inventory.json and form-config.json.
    """
    session_id = str(uuid.uuid4())
    session_form_path = os.path.join(SESSION_FORMS_DIR, f"{session_id}.json")

    # Check if the session file already exists (unlikely but defensive)
    while os.path.exists(session_form_path):
        session_id = str(uuid.uuid4())
        session_form_path = os.path.join(SESSION_FORMS_DIR, f"{session_id}.json")

    # Paths to data file
    data_inventory_path = os.path.join(FORM_DATA_DIR, "data-inventory.json")

    # Load data
    with open(data_inventory_path, "r", encoding="utf-8") as f:
        data_inventory = json.load(f)

    session_form_data = {form_type: data_inventory.get(form_type, [])}

    # Save session data
    with open(session_form_path, "w", encoding="utf-8") as f:
        json.dump(session_form_data, f, indent=4)

    return session_id, session_form_data

# ---------------------
# Load & Save Session Files
# ---------------------
def load_session_form(session_id: str) -> dict:
    """
    Loads the session-specific JSON form.
    Raises FileNotFoundError if the session file doesn't exist.
    """
    session_form_path = os.path.join(SESSION_FORMS_DIR, f"{session_id}.json")
    if not os.path.exists(session_form_path):
        raise FileNotFoundError(f"Session form for {session_id} not found.")

    with open(session_form_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_session_form(session_id: str, session_form: dict):
    """
    Saves the updated session-specific JSON form.
    """
    session_form_path = os.path.join(SESSION_FORMS_DIR, f"{session_id}.json")
    with open(session_form_path, "w", encoding="utf-8") as f:
        json.dump(session_form, f, indent=4)

# ---------------------
# Classification & Generation
# ---------------------
def classify_prompt(prompt: str) -> str:
    """
    Classifies the user input as 'Data Extraction' or 'Detailed Response' using T5.
    """
    try:
        # Format your classification prompt using the decision_prompt template
        decision_input = decision_prompt.format(user_input=prompt)

        # We’ll give T5 a "classification instruction" prefix, e.g. "classify: ... "
        # so T5 knows this is a classification task, not extraction.
        input_text = f"classify: {decision_input}"

        # Tokenize & generate
        inputs = t5_tokenizer.encode(input_text, return_tensors="pt")
        outputs = t5_model.generate(inputs, max_length=50, num_beams=5)
        decision = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("DECISION: " + decision)
        return decision.strip()
    except Exception as e:
        return f"Error in classification: {str(e)}"

# ---------------------
# Output Cleanse
# ---------------------
def output_cleanse(text: str) -> str:
    """
    Cleanses the output text by:
    - Trimming the last sentence if incomplete.
    - Removing unnecessary or misleading keywords and filler phrases.
    """
    keywords = [
        "What happens if I ", "Learn more about", "However,", 
        "For example,", "For more information,"
    ]
    
    # Convert text and keywords to lowercase for case-insensitive matching
    lower_text = text.lower()

    # Trim last sentence: Remove everything after the last period (".") if found
    match = re.search(r'^(.*?\.)[^.]*$', text, re.DOTALL)
    cleaned_text = match.group(1).strip() if match else text  # Return trimmed text or original if no period found

    # Create regex pattern to match each keyword (case-insensitive)
    pattern = r'(?i)(' + '|'.join(re.escape(kw.lower()) for kw in keywords) + r').*'

    # Find match position and remove everything after matched keyword
    match = re.search(pattern, lower_text)
    if match:
        cleaned_text = cleaned_text[:match.start()].strip()

    return cleaned_text

# ---------------------
# GPT Generator
# ---------------------
def generate_with_gpt(question: str, field: str) -> str:
    """
    Generates a detailed response using GPT-2 
    with a Question: / Answer: style prompt.
    """
    print(question + " Regarding " + field)
    prompt_text = f"<|startoftext|>Question: {question + ' Regarding ' + field}\nAnswer:"
    
    try:
        inputs = gpt_tokenizer(prompt_text, return_tensors="pt")
        outputs = gpt_model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=60,
            min_length=30,
            do_sample=True,
            top_k=50,
            top_p=0.85,
            temperature=0.7,
            pad_token_id=gpt_tokenizer.eos_token_id,
            repetition_penalty=1.1,  # discourage repetition
            no_repeat_ngram_size=2,  # block 2-gram repeats
            early_stopping=False
        )

        # Decode the tokens
        generated_text = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)

        if "Answer:" in generated_text:
            # Keep only text after "Answer:"
            generated_text = generated_text.split("Answer:", 1)[-1].strip()

        # Apply Output Cleanse
        return output_cleanse(generated_text)

    except Exception as e:
        return f"Error generating GPT response: {str(e)}"

# ---------------------
# T5 Generator
# ---------------------
def generate_with_t5(prompt: str) -> str:
    """
    Extracts data using T5.
    """
    try:
        # For extraction, we use a different prefix, e.g. "extract:"
        input_text = f"extract: {prompt}"
        inputs = t5_tokenizer.encode(input_text, return_tensors="pt")
        outputs = t5_model.generate(inputs, max_length=50, num_beams=5)
        t5_output = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Ensure JSON format if it looks like a key:value pair but isn't wrapped
        if ":" in t5_output and not t5_output.startswith("{"):
            t5_output = "{" + t5_output.strip() + "}"
        return t5_output
    except Exception as e:
        return f"Error generating T5 response: {str(e)}"

# ---------------------
# Main Prompt Handler
# ---------------------
def handle_prompt(prompt: str, session_form: dict, field: str):
    """
    Routes the prompt to the appropriate model (T5 for data extraction or GPT-2 for detailed response).
    `field` is the key that tells us which extraction prefix to use if 'Data Extraction' is required.
    """
    # Step 1: Classify with T5 or your chosen classifier
    decision = classify_prompt(prompt)  # e.g. "Data Extraction" or "Detailed Response"

    # Step 2: If T5 says "Data Extraction," prepend the relevant extraction prompt
    if "Data Extraction" in decision:
        # Look up the extraction command based on the field parameter
        # Fallback to a generic "extract:" if the field isn’t found
        extraction_prefix = field_commands.get(field.lower(), "extract: ")
        
        # Combine prefix + user prompt
        t5_input = extraction_prefix + prompt
        print(t5_input)
        # Generate with T5
        t5_output = generate_with_t5(t5_input)
        try:
            extracted_fields = json.loads(t5_output)

            # Validate that the extracted fields match the expected field keys
            invalid_fields = [key for key in extracted_fields.keys() if key.lower() not in field_commands]

            if invalid_fields:
                return f"Sorry! I didn't quite catch your command, let's try again!\n\nUnexpected output: {t5_output}", session_form

            # If everything is valid, update the form
            update_form(session_form, extracted_fields)
            print("FIELDS:", extracted_fields)
            return "Thank you, answer has been extracted for: " + field, session_form

        except json.JSONDecodeError:
            return f"Error: Invalid JSON output. T5 said:\n{t5_output}", session_form

    elif "Detailed Response" in decision:
        # For a detailed response, use GPT
        answer = generate_with_gpt(prompt, field)
        return answer, session_form

    else:
        # If classification is unclear, default to data extraction or handle differently
        extraction_prefix = field_commands.get(field.lower(), "extract: ")
        t5_input = extraction_prefix + prompt
        t5_output = generate_with_t5(t5_input)
        try:
            extracted_fields = json.loads(t5_output)
            update_form(session_form, extracted_fields)
            return json.dumps(extracted_fields, indent=4), session_form
        except json.JSONDecodeError:
             return f"Error: Unable to classify or parse JSON. T5 said: {t5_output}", session_form

# ---------------------
# Update Session Form
# ---------------------
def update_form(form_data, extracted_fields):
    """
    Update the session form data with the fields extracted by T5.
    We assume:
      form_data["Characteristics"] is a list of characteristic dicts, each with "Name", "Value", etc.
      form_data["Form"] (if present) is an object keyed by field name with a "SelectedValue".
    """
    characteristics = form_data.get("Characteristics", [])
    for key, value in extracted_fields.items():
        updated_in_characteristics = False
        # Try to match each extracted field to the "Characteristics" array
        for characteristic in characteristics:
            if characteristic["Name"].lower() == key.lower():
                characteristic["Value"] = value
                updated_in_characteristics = True
                break

        # If not found in "Characteristics", try the "Form" object
        if not updated_in_characteristics:
            frm = form_data.get("Form", {})
            if key in frm:
                frm[key]["SelectedValue"] = value

    return form_data
