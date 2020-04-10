import numpy as np
import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout


class MultilayerNN(Module):
    def __init__(self, sample_size, hidden_dim):
        super(MultilayerNN, self).__init__()
        self.sample_size = np.prod(sample_size)
        self.fc1 = Linear(self.sample_size, hidden_dim)
        self.fc2 = Linear(hidden_dim, hidden_dim//2)
        self.fc3 = Linear(hidden_dim//2, hidden_dim//4)
        self.fc4 = Linear(hidden_dim//4, hidden_dim//8)
        self.fc5 = Linear(hidden_dim//8, hidden_dim//16)
        self.fc6 = Linear(hidden_dim//16, 1)
        self.dropout = Dropout(p=0.5)

    def forward(self, txs):
        x = self.dropout(F.relu(self.fc1(txs)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.dropout(F.relu(self.fc5(x)))
        return self.fc6(x)
