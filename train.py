import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from models.disease_model import DiseaseClassifier

class LabDataset(Dataset):
    def __init__(self, features, labels):
        # Assume features are already preprocessed (no scaling inside this class)
        self.X = features
        self.y = labels

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# Load and preprocess data
with open("data/synthetic_lab_reports.json", "r") as f:
    data = json.load(f)

# Extract features and labels
features = []
labels = []
for entry in data:
    gender = 1 if entry['gender'] == 'male' else 0
    features.append([
        entry['age'],
        gender,
        entry['glucose'],
        entry['cholesterol'],
        entry['bp_systolic'],
        entry['creatinine']
    ])
    labels.append(entry['disease'])

# Split into train/val/test
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)

# Initialize and fit scaler on training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Create datasets (features are already scaled)
train_dataset = LabDataset(X_train, y_train)
val_dataset = LabDataset(X_val, y_val)
test_dataset = LabDataset(X_test, y_test)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# Initialize model, loss, and optimizer
model = DiseaseClassifier(input_size=6)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {correct/total:.4f}")

# Save the model and scaler
torch.save(model.state_dict(), "models/disease_model.pt")
torch.save(scaler, "models/scaler.pt")