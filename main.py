import torch
import torch.nn as nn
import torch.optim as optim

from src.data.data_loader import get_data_loader
from src.model.etm import ETM
from src.config.config import DEVICE, EPOCHS, LEARNING_RATE

def train():
    loader = get_data_loader()
    model = ETM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(1, EPOCHS+1):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred, theta, beta = model(batch)
            loss = criterion(pred.log(), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch:02d} — Avg. Loss: {avg_loss:.4f}")

    # In top từ
    beta = model.get_beta().cpu().detach().numpy()
    for k in range(beta.shape[0]):
        top10 = beta[k].argsort()[-10:][::-1]
        print(f"Topic {k:02d}: {top10.tolist()}")

if __name__ == "__main__":
    train()