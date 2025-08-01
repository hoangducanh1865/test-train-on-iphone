import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from src.config.config import VOCAB_SIZE, BATCH_SIZE

class BowDataset(Dataset):
    def __init__(self, bows):
        self.docs = torch.tensor(bows.toarray(), dtype=torch.float32)
        self.docs = self.docs / (self.docs.sum(dim=1, keepdim=True) + 1e-9)  # normalize

    def __len__(self):
        return self.docs.shape[0]

    def __getitem__(self, idx):
        return self.docs[idx]

def get_data_loader():
    # 1. Load text data
    data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes')).data

    # 2. Convert text to Bag-of-Words (BOW) representation
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, stop_words='english')
    bows = vectorizer.fit_transform(data)

    # 3. Dataset & DataLoader
    dataset = BowDataset(bows)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)