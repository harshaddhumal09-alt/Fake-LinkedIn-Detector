from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model/fake_linkedin_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = [
        float(request.form["connections"]),
        float(request.form["skills"]),
        float(request.form["experience"]),
        float(request.form["recommendations"]),
        float(request.form["posts"]),
        float(request.form["profile_completion"])
    ]

    prediction = model.predict([data])[0]

    prediction = model.predict([data])[0]

    probability = model.predict_proba([data])[0]
    confidence = round(max(probability) * 100, 2)

    result = "Genuine Profile" if prediction == 1 else "Fake Profile"

    if confidence >= 80:
        risk = "Low Risk"
    elif confidence >= 50:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    from flask import render_template
    result = "Genuine Profile" if prediction == 1 else "Fake Profile"
    if result == "Fake Profile":
        risk = "High Risk"
    else:
        risk = "Low Risk"

    return f"""
    <h1>Prediction: {result}</h1>
    <h2>Confidence: {confidence}%</h2>
    <h3>Risk Level: {risk}</h3>
    <br>
    <a href="/">Go Back</a>
    """

if __name__ == "__main__":
    app.run(debug=True)