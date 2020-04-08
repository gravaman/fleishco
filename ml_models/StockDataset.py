from os.path import splitext, basename
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml_models.utils import list_files


class StockDataset(Dataset):
    def __init__(self, root_dir, idxs=None, index_col=None,
                 T_back=30, T_fwd=10, mu=None, std=None):
        # init params
        self.ticker_list = list_files(root_dir)
        if idxs is not None:
            self.ticker_list = list(np.array(self.ticker_list)[idxs])

        self.tickers = [splitext(basename(p))[0] for p in self.ticker_list]
        self.index_col = index_col
        self.T_back = T_back
        self.T_fwd = T_fwd
        self.T = T_back+T_fwd
        self.mu = mu
        self.std = std

    def size(self):
        """
        shape of X and y
        """
        df = pd.read_csv(self.ticker_list[0], nrows=10,
                         index_col=self.index_col)
        N = self.__len__()
        D = df.shape[1]-1
        return (N, self.T_back, D), (N, self.T_fwd)

    def get_stats(self):
        """
        Calculates the mean and std of the dataset excluding
        the forward period. Causes data to be normalized upon
        future retrieval by dataloader.

        return
        N (scalar): number of samples
        mu (D,): mean of data features (includes target)
        std (D,): std of data features (includes target)
        """
        # mu: sum features/total samples
        xsize, _ = self.size()
        N, totals = 0, np.zeros(xsize[2]+1)
        for ticker_path in self.ticker_list:
            data = pd.read_csv(
                ticker_path,
                index_col=self.index_col
            ).dropna().sort_values(
                ['Date'],
                ascending=[True]
            ).values.astype('float')

            # sum except for last T_fwd records
            data = data[:-self.T_fwd, :]
            N += data.shape[0]
            totals += data.sum(axis=0)
        mu = totals/N

        # std sqrt SS/N
        ss = np.zeros(mu.shape)
        for ticker_path in self.ticker_list:
            data = pd.read_csv(
                ticker_path,
                index_col=self.index_col
            ).dropna().sort_values(
                ['Date'],
                ascending=[True]
            ).values.astype('float')

            # sum squares except last T_fwd records
            data = data[:-self.T_fwd, :]
            ss += ((data-mu)**2).sum(axis=0)
        std = (ss/N)**0.5

        # store for item retrieval
        self.mu = mu
        self.std = std

        return mu, std

    def __len__(self):
        return len(self.ticker_list)

    def __getitem__(self, ticker_idx):
        # setup for data loading
        if torch.is_tensor(ticker_idx):
            ticker_idx = ticker_idx.tolist()

        file_path = self.ticker_list[ticker_idx]

        # load data
        data = pd.read_csv(
            file_path,
            index_col=self.index_col
        ).dropna().sort_values(
            ['Date'],
            ascending=[True]
        ).values.astype('float')

        # conditionally standardize
        if self.mu is not None and self.std is not None:
            data = (data - self.mu)/self.std

        # convert data into time series (truncate beginning)
        # data: (TS_CNT, D_in, T)
        N, D_in = data.shape
        TS_CNT = N // self.T
        data = np.array(np.split(data[N % self.T:],
                                 TS_CNT)).reshape((TS_CNT, self.T, -1))

        # convert to time series
        X, y = data[:, :self.T_back, :-1], data[:, self.T_back:, -1]
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return X, y


class StockBatch:
    mbatch_size = 50  # samples per batch

    def __init__(self, batch):
        self.X, self.y = [torch.cat(b, dim=0) for b in zip(*batch)]
        idxs = np.random.choice(len(self.X), self.mbatch_size)
        self.X, self.y = self.X[idxs], self.y[idxs]

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.y = self.y.pin_memory()
        return self
