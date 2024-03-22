from utils import generate_sample
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.functional import one_hot
import Levenshtein as L

class Random_seq_seq_d_dataset(Dataset):
    def __init__(self, seq_length, padded_length, seed=7, num_samples=1000000):
        super(Random_seq_seq_d_dataset, self).__init__()
        self.seq_length = seq_length
        self.padded_length = padded_length
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
    def __len__(self):
        return self.num_samples
    def __getitem__(self,idx):
        if np.random.rand()>0.5:
            seq_a = np.random.randint(0,5,size=(self.seq_length,))
            seq_b = np.random.randint(0,5,size=(self.seq_length,))
            distance = L.distance(seq_a,seq_b)
        else:
            n = np.random.randint(1,5)
            seq_a,seq_b = generate_sample(self.seq_length, self.padded_length, distance=n)
            distance = n
        if distance != 1:
            distance = 2
        seq_a = seq_a[:self.padded_length]
        seq_b = seq_b[:self.padded_length]
        seq_a = np.concatenate([seq_a,[4]*(self.padded_length-len(seq_a))]).astype(np.int64)
        seq_b = np.concatenate([seq_b,[4]*(self.padded_length-len(seq_b))]).astype(np.int64)
        seq_a = one_hot(torch.tensor(seq_a),num_classes=5).numpy().astype(np.float32).T
        seq_b = one_hot(torch.tensor(seq_b),num_classes=5).numpy().astype(np.float32).T
        
        return np.array([seq_a,seq_b]), distance