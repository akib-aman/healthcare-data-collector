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

# ---------------------
# Setup and initalise session with client
# ---------------------
@app.route("/api/setup", methods=["POST"])
def create_session():
    """
    Creates a new session, generates session ID, and returns the initial form data.
    """
    data = request.get_json()
    form_type = data.get("formType", "")

    if not form_type:
        return jsonify({"error": "formType is required"}), 400

    # Create session and get form data
    session_id, form_data = create_session_form(form_type)

    return jsonify({"session_id": session_id, "form_data": form_data})

# ---------------------
# Handle user prompts
# ---------------------
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

# ---------------------
# Conclude User Sessions
# ---------------------
@app.route("/api/finish", methods=["POST"])
def finish_session():
    data = request.get_json()
    session_id = data.get("session_id")

    if not session_id:
        return jsonify({"error": "Missing session_id."}), 400

    # 1. Load the session form once
    try:
        session_form = load_session_form(session_id)
    except FileNotFoundError:
        return jsonify({"error": f"Session {session_id} not found."}), 404

    # Send to Records: simulate sending the session form JSON data to a medical database or GP database.
    # For simulation, we log the session form to the console.
    print("Sending session form to records:", session_form)

    # 2. Delete the entire session JSON file
    session_form_path = os.path.join(SESSION_FORMS_DIR, f"{session_id}.json")
    try:
        os.remove(session_form_path)
    except Exception as e:
        return jsonify({"error": f"Error deleting session form: {str(e)}"}), 500

    return jsonify({"message": "Session finished and form data sent to records."})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
