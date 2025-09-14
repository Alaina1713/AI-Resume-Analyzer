import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import pandas as pd
from dataset import ResumeDataset
from model import ResumeAnalyzer
from preprocess import clean_text, tokenize

# Load sample dataset
data = pd.read_csv('../data/sample_resumes.csv')

texts = data['resume_text'].tolist()
labels = data.drop(columns=['resume_text']).values.tolist()

# Create PyTorch dataset
dataset = ResumeDataset(texts, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
num_labels = len(labels[0])
model = ResumeAnalyzer(num_labels=num_labels)

# Optimizer and loss
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()  # Multi-label classification

# Training loop
for epoch in range(5):  # small demo training
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# Save trained weights
torch.save(model.state_dict(), '../models/model_state.pth')
print("Model saved to '../models/model_state.pth'")
