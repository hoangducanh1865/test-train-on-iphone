import matplotlib.pyplot as plt

def plot_training_curves(losses, coherence, diversity):
    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(10, 6))

    plt.plot(epochs, losses, label="Loss")
    plt.plot(epochs, coherence, label="Coherence")
    plt.plot(epochs, diversity, label="Diversity")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Training Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()