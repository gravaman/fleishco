import torch
import pandas as pd
import numpy as np
from os.path import join
from torch.utils.data import Dataset


class CreditDataset(Dataset):
    def __init__(self, batch_path, root_dir, transform=None,
                 minibatch_size=256, val_split=0.2,
                 test_split=0.1, split='train', rs=0):
        batches = pd.read_csv(batch_path, header=None)
        np.random.seed(rs)
        train_split = 1-test_split-val_split
        
        total_cnt = len(batches)
        train_cnt = int(np.floor(total_cnt*train_split))
        val_cnt = int(np.floor(total_cnt*val_split))
        test_cnt = total_cnt-train_cnt-val_cnt
        
        if split == 'train':
            idxs = np.random.choice(total_cnt, train_cnt, replace=False)
        elif split == 'val':
            idxs = np.random.choice(total_cnt, val_cnt, replace=False)
        elif split == 'test':
            idxs = np.random.choice(total_cnt, test_cnt, replace=False)
        else:
            raise ValueError(f'unknown split key: {split}')

        self.split = split
        self.batches = batches.iloc[idxs]
        self.root_dir = root_dir
        self.transform = transform
        self.minibatch_size = minibatch_size

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load credit transactions file
        bp = join(self.root_dir, self.batches.iloc[idx, 0])
        txs = pd.read_csv(bp, dtype=np.float64)

        # sample minibatch
        idxs = np.random.randint(len(txs), size=self.minibatch_size)
        sample = (txs.iloc[idxs, :-1].values, txs.iloc[idxs, -1].values)

        if self.transform:
            sample = self.transform(sample)

        return sample


class Standardize:
    def __init__(self, params):
        self.x_mu, self.x_std = params[0, :-1], params[1, :-1]
        self.y_mu, self.y_std = params[0, -1], params[1, -1]

    def __call__(self, sample):
        x, y = sample
        x_out, y_out = (x-self.x_mu)/self.x_std, (y-self.y_mu)/self.y_std
        return (x_out, y_out)


class ToTensor:
    def __call__(self, sample):
        x, y = sample
        return (torch.from_numpy(x), torch.tensor(y))