import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load dataset
data = pd.read_csv("dataset/linkedin_profiles.csv")

# Convert labels
data["label"] = data["label"].map({"Fake": 0, "Genuine": 1})

# Features and target
X = data.drop("label", axis=1)
y = data["label"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open("model/fake_linkedin_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")