from flask import Flask, request, jsonify
from flask_cors import CORS
from prompt_inputs import handle_prompt  # Import your logic here

app = Flask(__name__)
CORS(app)  # Enable CORS for communication with the React frontend

@app.route("/api/prompt", methods=["POST"])
def handle_prompt_route():
    """
    Handle incoming prompts from the React frontend.
    """
    data = request.json
    user_prompt = data.get("prompt", "")
    
    # Use the handle_prompt function from prompt-inputs.py
    response = handle_prompt(user_prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
