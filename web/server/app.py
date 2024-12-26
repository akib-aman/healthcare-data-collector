from flask import Flask, request, jsonify
from flask_cors import CORS

# For demonstration, import your model code or relevant modules here.
# Example:
# from ... import load_gpt_model, load_t5_model  # adjust import paths as needed

app = Flask(__name__)
CORS(app)  # Enable CORS so your React frontend can talk to Flask

# Load your GPT or T5 model once (optional)
# gpt_tokenizer, gpt_model = load_gpt_model()
# t5_tokenizer, t5_model = load_t5_model()

@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    """
    Expects a JSON object like:
    { "prompt": "User's question or statement" }
    """
    data = request.json
    user_prompt = data.get("prompt", "")

    # Here’s where you route to GPT or T5, or do simple classification:
    # For now, let's assume we just want GPT to handle everything:
    #
    # answer = run_inference_gpt(gpt_tokenizer, gpt_model, user_prompt)
    # or do a mock response for testing:
    answer = f"Echo from server: {user_prompt}"

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
