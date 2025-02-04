import os
from gensim.models import Word2Vec
from flask import Flask, render_template, request

# app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')

# Load Word2Vec model from 'models' folder
def load_model():
    model_path = "models/word2vec.model"

    # Check if the file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found at: {model_path}")
        return None

    try:
        model = Word2Vec.load(model_path)
        print("✅ Word2Vec Model loaded successfully!")
        return model
    except Exception as e:
        import traceback
        traceback.print_exc()  # Shows detailed error
        print(f"❌ Error loading model: {e}")
        return None

# Initialize the model
model = load_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/similarity", methods=["POST"])
def similarity():
    if model is None:
        return render_template("index.html", result="❌ Model is not loaded properly. Please check the model file.")
    
    word1 = request.form.get("word1")
    word2 = request.form.get("word2")

    try:
        similarity_score = model.wv.similarity(word1, word2)
        result = f"Similarity between '{word1}' and '{word2}': {similarity_score:.4f}"
        print(result)
        return render_template("index.html", result=result)  # ✅ Render template with the result
    except KeyError as e:
        error_message = f"❌ Error: {e} (One or both words not in vocabulary)"
        print(error_message)
        return render_template("index.html", result=error_message)

if __name__ == "__main__":
    # app.run(debug=True)
    app.run(debug=True, port=5001)
