import json
import torch
import numpy as np
from models.disease_model import DiseaseClassifier

# Load model and scaler
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiseaseClassifier(input_size=6)
model.load_state_dict(torch.load("models/disease_model.pt", map_location=device))
model.eval()
scaler = torch.load("models/scaler.pt")

# Load new JSON data (example)
new_data = {
    "age": 55,
    "gender": "female",
    "glucose": 130,
    "cholesterol": 220,
    "bp_systolic": 145,
    "creatinine": 1.5
}

# Preprocess
gender = 1 if new_data['gender'] == 'male' else 0
features = np.array([
    new_data['age'],
    gender,
    new_data['glucose'],
    new_data['cholesterol'],
    new_data['bp_systolic'],
    new_data['creatinine']
])
features = scaler.transform([features])
features = torch.tensor(features, dtype=torch.float32).to(device)

# Predict
with torch.no_grad():
    output = model(features)
    _, predicted = torch.max(output, 1)
    disease_labels = ["Healthy", "Diabetes", "Hypertension"]
    print(f"Predicted Disease: {disease_labels[predicted.item()]}")