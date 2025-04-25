
from flask import Flask, request, jsonify
from run4 import main  # this is now possible since main() is defined

app = Flask(__name__)

@app.route("/api/research", methods=["GET", "POST"])
def research():
    question = request.args.get("question") or request.form.get("question")

    if not question:
        return jsonify({"status": "error", "message": "Missing 'question' parameter"}), 400

    result = main(question)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
