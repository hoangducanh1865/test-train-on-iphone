
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TOPICS = 20
EMBED_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 1e-2
VOCAB_SIZE = 5000  # giới hạn từ vựng để đơn giản hóa