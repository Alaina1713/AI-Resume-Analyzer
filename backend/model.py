import torch
import torch.nn as nn
from transformers import AutoModel

class ResumeAnalyzer(nn.Module):
    def __init__(self, pretrained_model='bert-base-uncased', num_labels=10):
        super(ResumeAnalyzer, self).__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # CLS token
        dropped = self.dropout(pooled_output)
        logits = self.classifier(dropped)
        return logits
