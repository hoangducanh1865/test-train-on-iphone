import matplotlib.pyplot as plt


def split_text_word(texts: list[str]) -> list[list[str]]:
    return [text.split() for text in texts]

def plot_training_curves(losses, coherence, diversity, prefix="training_plot"):
    epochs = list(range(1, len(losses) + 1))

    # Plot Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, label="Loss", color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_loss.png")
    plt.close()

    # Plot Coherence
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, coherence, label="Coherence", color='green')
    plt.xlabel("Epoch")
    plt.ylabel("Topic Coherence")
    plt.title("Topic Coherence over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_coherence.png")
    plt.close()

    # Plot Diversity
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, diversity, label="Diversity", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Topic Diversity")
    plt.title("Topic Diversity over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{prefix}_diversity.png")
    plt.close()