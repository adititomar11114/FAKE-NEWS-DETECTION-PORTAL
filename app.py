from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/fake_news_model.pkl")

# Load TF-IDF vectorizer
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    news_text = request.form.get("news_text", "").strip()

    if not news_text:
        return render_template(
            "index.html",
            prediction="Please enter some news text.",
            confidence=None
        )

    # Convert news text into TF-IDF features
    text_vector = vectorizer.transform([news_text])

    # Predict
    prediction = model.predict(text_vector)[0]

    # Prediction probability
    probabilities = model.predict_proba(text_vector)[0]

    confidence = max(probabilities) * 100

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(confidence, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)
