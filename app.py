from flask import Flask, render_template, request, jsonify
from rag_logic import get_response
import logging

logging.basicConfig(filename='xatbot.log', level=logging.INFO, format='%(asctime)s - %(message)s')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    response = get_response(user_input)
    logging.info(f"Input: {user_input} | Response: {response}")
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)