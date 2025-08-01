import torch
import torch.nn as nn
import torch.optim as optim

from src.data.data_loader import get_data_loader
from src.model.etm import ETM
from src.config.config import DEVICE, EPOCHS, LEARNING_RATE

from src.eva.topic_coherence import compute_coherence
from src.eva.topic_diversity import compute_diversity

def train():
    loader, vectorizer = get_data_loader(return_vectorizer=True)
    vocab = vectorizer.get_feature_names_out()

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

    # In top words
    beta = model.get_beta().cpu().detach().numpy()
    all_topics = []
    for k in range(beta.shape[0]):
        top_indices = beta[k].argsort()[-10:][::-1]
        words = [vocab[i] for i in top_indices]
        print(f"Topic {k:02d}: {' | '.join(words)}")
        all_topics.append(words)

    # Evaluate
    print("\n--- Evaluation ---")
    print(f"Topic Diversity: {compute_diversity(all_topics):.4f}")
    print(f"Topic Coherence (approx): {compute_coherence(all_topics, vectorizer):.4f}")

if __name__ == "__main__":
    train()