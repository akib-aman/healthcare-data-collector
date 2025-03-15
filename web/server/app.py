from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from prompt_inputs import handle_prompt, create_session_form, load_session_form, save_session_form

app = Flask(__name__)
CORS(app)

# Directory for session JSON files
SESSION_FORMS_DIR = "form-data/session-forms"
os.makedirs(SESSION_FORMS_DIR, exist_ok=True)


@app.route("/api/session", methods=["POST"])
def create_session():
    """
    Creates a new session for a specific form type and returns the session ID.
    Expects JSON with { "formType": "<someFormType>" } in the POST body.
    """
    data = request.get_json()
    form_type = data.get("formType", "").lower() 

    # Pass the form_type to create_session_form
    session_id = create_session_form(form_type)
    return jsonify({"session_id": session_id})

@app.route("/api/prompt", methods=["POST"])
def handle_prompt_route():
    data = request.get_json()
    session_id = data.get("session_id")
    user_prompt = data.get("prompt", "")
    active_field = data.get("field", "")

    if not session_id:
        return jsonify({"error": "Missing session_id."}), 400

    # 1. Load the session form ONCE here
    try:
        session_form = load_session_form(session_id)
    except FileNotFoundError:
        return jsonify({"error": f"Session {session_id} not found."}), 404

    # 2. Handle the user prompt, passing the already-loaded form
    response, updated_form = handle_prompt(user_prompt, session_form, active_field)

    # 3. Save the updated session form
    save_session_form(session_id, updated_form)

    # 4. Return the response
    return jsonify({ "response": response, "updated_form": updated_form })

@app.route("/api/setup", methods=["POST"])
def setup_form():
    """
    Handles setup of a specific form type and returns the appropriate data.
    """
    # 1. Build absolute paths based on the directory of this app.py file
    base_dir = os.path.dirname(__file__)  # folder where app.py is located
    data_inventory_path = os.path.join(base_dir, "form-data", "data-inventory.json")
    form_config_path = os.path.join(base_dir, "form-data", "form-config.json")

    # 2. Read JSON from POST body
    request_data = request.get_json()
    form_type = request_data.get("formType", "").lower()

    # 3. Load the relevant JSON files
    with open(data_inventory_path, 'r', encoding='utf-8') as f:
        data_inventory = json.load(f)

    with open(form_config_path, 'r', encoding='utf-8') as f:
        form_config = json.load(f)

    # 4. Use the form_config to determine which data to send back
    form_def = form_config["forms"].get(form_type)

    if not form_def:
        # If no matching form definition, return an error
        return jsonify({"error": "Unknown formType requested."}), 400

    if form_def["type"] == "CHARACTERISTICS_ONLY":
        return jsonify(data_inventory["Characteristics"])
    elif form_def["type"] == "FULL_FORM":
        return jsonify(data_inventory)
    else:
        # Handle other variants as needed
        return jsonify(data_inventory)  # or something else

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
