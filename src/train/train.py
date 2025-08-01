import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.data.data_loader import get_data_loader
from src.model.etm import ETM
from src.config.config import DEVICE, EPOCHS, LEARNING_RATE
from src.eva.topic_coherence import compute_coherence
from src.eva.topic_diversity import compute_diversity
from src.utils.utils import plot_training_curves
from src.utils.utils import split_text_word
from IPython.display import Image, display
import os


def train_etm():
    loader, vectorizer = get_data_loader(return_vectorizer=True)
    raw_texts = vectorizer._args["input_data"]
    vocab = vectorizer.get_feature_names_out().tolist()

    model = ETM().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.KLDivLoss(reduction='batchmean')

    train_losses = []
    coherence_scores = []
    diversity_scores = []

    best_coherence = float("-inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        prog_bar = tqdm(loader, desc=f"Epoch {epoch:02d}", leave=False)
        for batch in prog_bar:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            pred, _, _ = model(batch)
            loss = criterion(pred.log(), batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)

        avg_loss = total_loss / len(loader.dataset)
        train_losses.append(avg_loss)

        # Evaluate topics
        beta = model.get_beta().cpu().detach().numpy()
        topics = [
            [vocab[i] for i in beta[k].argsort()[-10:][::-1] if vocab[i] in set(' '.join(raw_texts).split())]
            for k in range(beta.shape[0])
        ]

        # Coherence (c_v) and Diversity
        coherence = compute_coherence(
            reference_corpus=raw_texts,
            vocab=vocab,
            top_words=topics,
            coherence_type='c_v'
        )
        diversity = compute_diversity(topics)

        coherence_scores.append(coherence)
        diversity_scores.append(diversity)

        print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Coherence: {coherence:.4f} | Diversity: {diversity:.4f}")

        # Early stopping
        if coherence > best_coherence:
            best_coherence = coherence
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print("\nFinal topics:")
    for i, topic in enumerate(topics):
        print(f"Topic {i:02d}: {' | '.join(topic)}")

