from os.path import splitext, basename
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from ml_models.utils import list_files


class TickerDataset(Dataset):
    def __init__(self, ticker_path, index_col=None,
                 T_back=10, T_fwd=1, stats=None, standardize=True):
        super(Dataset, self).__init__()

        self.ticker_path = ticker_path
        self.index_col = index_col
        self.T_back = T_back
        self.T_fwd = T_fwd
        self.T = T_back+T_fwd

        data = pd.read_csv(
            ticker_path,
            index_col=index_col,
        ).dropna().sort_values(
            ['Date'],
            ascending=[True]
        )

        # skip beginning stub
        data = data.iloc[data.shape[0] % self.T:]

        # store first fwd dates for plotting
        self.fwd_dates = pd.to_datetime(
            data[self.T_back::self.T].index.values)

        # store first row for indexing data
        self.base_index = data.iloc[0][None, None, :]
        data = data.values.astype('float')

        # convert data to time series format
        # data: (TS_CNT, D_in, T)
        TS_CNT = data.shape[0] // self.T
        data = np.array(np.split(
            data,
            TS_CNT
        )).reshape((TS_CNT, self.T, -1))

        # index first period
        data = data/self.base_index

        # conditionally standardize
        self.stats = stats
        if stats is not None and standardize:
            mu, std = stats
            data = (data - mu)/std
        elif standardize:
            mu = np.mean(data, axis=0)
            std = np.mean(data, axis=0)
            self.stats = (mu, std)
            data = (data-mu)/std

        # convert to input and target tensors
        self.X, self.y = data[:, :self.T_back, :], data[:, self.T_back:, -1]
        self.X = torch.tensor(self.X, dtype=torch.float).unsqueeze(0)
        self.y = torch.tensor(self.y, dtype=torch.float).unsqueeze(0)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx], self.y[idx]


class StockDataset(Dataset):
    def __init__(self, root_dir, idxs=None, index_col=None,
                 T_back=30, T_fwd=10, mu=None, std=None):
        # init params
        self.ticker_list = list_files(root_dir)
        if idxs is not None and len(idxs) > 0:
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
        D = df.shape[1]
        return (N, self.T_back, D), (N, self.T_fwd)

    def get_stats(self):
        """
        Calculates the mean and std of the dataset excluding
        the forward period. Causes data to be normalized upon
        future retrieval by dataloader.

        return
        N (int): number of samples
        mu (D,): mean of data features (includes target)
        std (D,): std of data features (includes target)
        """
        # mu: sum features/total samples
        xsize, _ = self.size()
        N, totals = 0, np.zeros(xsize[2])
        for i, ticker_path in enumerate(self.ticker_list):
            data = pd.read_csv(
                ticker_path,
                index_col=self.index_col
            ).dropna().sort_values(
                ['Date'],
                ascending=[True]
            )
            data = data.iloc[data.shape[0] % self.T:]

            # index based on initial period
            data = data/data.iloc[0]
            data = data.values.astype('float')

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
            )

            data = data.iloc[data.shape[0] % self.T:]

            # index based on initial period
            data = data/data.iloc[0]
            data = data.values.astype('float')

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
        )

        # index based on initial period
        data = data.iloc[data.shape[0] % self.T:]
        data = data/data.iloc[0]
        data = data.values.astype('float')

        # conditionally standardize
        if self.mu is not None and self.std is not None:
            data = (data - self.mu)/self.std

        # convert data into time series (truncate beginning)
        # data: (TS_CNT, D_in, T)
        TS_CNT = data.shape[0] // self.T
        data = np.array(np.split(data,
                                 TS_CNT)).reshape((TS_CNT, self.T, -1))

        # convert to time series
        X, y = data[:, :self.T_back, :], data[:, self.T_back:, -1]
        X = torch.tensor(X, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        return X, y


class TickerBatch:
    def __init__(self, batch):
        self.X, self.y = [torch.float(b) for b in zip(*batch)]

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.y = self.y.pin_memory()
        return self


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
