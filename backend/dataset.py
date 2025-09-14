import torch
from torch.utils.data import Dataset
from preprocess import clean_text, tokenize

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = clean_text(self.texts[idx])
        tokens = tokenize(text)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'labels': label
        }
