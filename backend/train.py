import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import ResumeDataset
from model import ResumeAnalyzer

# Sample data
texts = ["Python developer with ML experience", "Data analyst proficient in SQL and Tableau"]
labels = [[1,0,0,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0,0,0]]  # Multi-label example

dataset = ResumeDataset(texts, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = ResumeAnalyzer(num_labels=10)
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(3):
    model.train()
    for batch in loader:
        optimizer.zero_grad()
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), '../models/model_state.pth')
