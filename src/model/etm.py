import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config.config import VOCAB_SIZE, NUM_TOPICS, EMBED_SIZE

class ETM(nn.Module):
    def __init__(self):
        super().__init__()
        self.rho   = nn.Parameter(torch.randn(VOCAB_SIZE, EMBED_SIZE) * 0.01)
        self.alpha = nn.Parameter(torch.randn(NUM_TOPICS, EMBED_SIZE) * 0.01)
        self.infer = nn.Linear(VOCAB_SIZE, NUM_TOPICS, bias=False)

    def get_beta(self):
        logits = self.alpha @ self.rho.t()  # K x V
        return F.softmax(logits, dim=1)

    def forward(self, bows):
        theta = F.softmax(self.infer(bows), dim=1)  # B x K
        beta = self.get_beta()                     # K x V
        preds = theta @ beta                       # B x V
        return preds, theta, beta