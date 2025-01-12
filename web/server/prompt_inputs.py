import os
import json
import uuid
from langchain.prompts import PromptTemplate
from transformers import GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration

# ---------------------
# Decision Prompt
# ---------------------
decision_prompt = PromptTemplate(
    input_variables=["user_input"],
    template="Does this input require data extraction or a detailed response? Input: {user_input}. Answer with either 'Data Extraction' or 'Detailed Response'."
)

# ---------------------
# Load Models & Tokenizers
# ---------------------
gpt_tokenizer = GPT2Tokenizer.from_pretrained("./gpt-trained-model")
gpt_model = GPT2LMHeadModel.from_pretrained("./gpt-trained-model")
t5_tokenizer = T5Tokenizer.from_pretrained("./t5-trained-model")
t5_model = T5ForConditionalGeneration.from_pretrained("./t5-trained-model")

# ---------------------
# Directory to store session-specific forms
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # The directory of the current script
FORM_DATA_DIR = os.path.join(BASE_DIR, "form-data")    # Path to "form-data"
SESSION_FORMS_DIR = os.path.join(FORM_DATA_DIR, "session-forms")  # Path to "form-data/session-forms"

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

    # Paths to data files
    data_inventory_path = os.path.join(FORM_DATA_DIR, "data-inventory.json")
    form_config_path = os.path.join(FORM_DATA_DIR, "form-config.json")

    # Load data
    with open(data_inventory_path, "r", encoding="utf-8") as f:
        data_inventory = json.load(f)
    with open(form_config_path, "r", encoding="utf-8") as f:
        form_config = json.load(f)

    # Process form type
    form_def = form_config["forms"].get(form_type.lower())
    session_form_data = data_inventory if not form_def else (
        {"Characteristics": data_inventory.get("Characteristics", [])} if form_def["type"] == "CHARACTERISTICS_ONLY"
        else data_inventory
    )

    # Save session data
    with open(session_form_path, "w", encoding="utf-8") as f:
        json.dump(session_form_data, f, indent=4)

    return session_id

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
    Classifies the user input as 'Data Extraction' or 'Detailed Response'.
    """
    try:
        decision_input = decision_prompt.format(user_input=prompt)
        inputs = gpt_tokenizer(decision_input, return_tensors="pt")
        outputs = gpt_model.generate(inputs.input_ids, max_length=50, num_beams=5)
        decision = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return decision.strip()
    except Exception as e:
        return f"Error in classification: {str(e)}"

def generate_with_gpt(prompt: str) -> str:
    """
    Generates a detailed response using GPT-2.
    """
    try:
        inputs = gpt_tokenizer(prompt, return_tensors="pt")
        outputs = gpt_model.generate(inputs.input_ids, max_length=50, num_beams=5)
        return gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Error generating GPT response: {str(e)}"

def generate_with_t5(prompt: str) -> str:
    """
    Extracts data using T5.
    """
    try:
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
def handle_prompt(prompt: str, session_form: dict):
    """
    Routes the prompt to the appropriate model, updates `session_form` in-memory, 
    and returns (response, updated_form).
    """
    decision = classify_prompt(prompt)

    if "Data Extraction" in decision:
        t5_output = generate_with_t5(prompt)
        try:
            extracted_fields = json.loads(t5_output)
            update_form(session_form, extracted_fields)  # in-memory update
            # Return the extracted fields as string plus the updated form
            return json.dumps(extracted_fields, indent=4), session_form
        except json.JSONDecodeError:
            return f"Error: Invalid JSON output. T5 said: {t5_output}", session_form

    elif "Detailed Response" in decision:
        # GPT response only; no form update
        answer = generate_with_gpt(prompt)
        return answer, session_form

    else:
        # If classification is unclear, try data extraction anyway
        t5_output = generate_with_t5(prompt)
        try:
            extracted_fields = json.loads(t5_output)
            update_form(session_form, extracted_fields)  # in-memory update
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
    print("FIELDS:", extracted_fields)
    # Update "Characteristics" if it exists
    characteristics = form_data.get("Characteristics", [])
    for key, value in extracted_fields.items():
        updated_in_characteristics = False
        for characteristic in characteristics:
            if characteristic["Name"].lower() == key.lower():
                characteristic["Value"] = value
                updated_in_characteristics = True
                break

        if not updated_in_characteristics:
            # If not found in "Characteristics", try the "Form" object
            frm = form_data.get("Form", {})
            if key in frm:
                frm[key]["SelectedValue"] = value

    return form_data
