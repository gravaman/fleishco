from os import listdir
from os.path import join, isfile, splitext, basename
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class StockDataset(Dataset):
    def __init__(self, root_dir, nrows=None, index_col=None,
                 T_back=30, T_fwd=10):
        # init params
        self.ticker_list = [join(root_dir, p) for p in listdir(root_dir)
                            if isfile(join(root_dir, p))]
        self.tickers = [splitext(basename(p))[0] for p in self.ticker_list]
        self.nrows = nrows
        self.index_col = index_col
        self.T_back = T_back
        self.T_fwd = T_fwd
        self.T = T_back+T_fwd

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
            nrows=self.nrows,
            index_col=self.index_col
        ).dropna().sort_values(
            ['Date'],
            ascending=[True]
        ).values.astype('float')

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
    def __init__(self, batch, mbatch_size=50):
        self.X, self.y = [torch.cat(b, dim=0) for b in zip(*batch)]
        idxs = np.random.choice(len(self.X), mbatch_size)
        self.X, self.y = self.X[idxs], self.y[idxs]

    def pin_memory(self):
        self.X = self.X.pin_memory()
        self.y = self.y.pin_memory()
        return self


def setup_logger(logger_name):
    # create and return logger with given name
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s '
                                  '| %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    logger = setup_logger('lstm')
    logger.info('Loading Stock Data Loader Tests')

    ticker_dir = 'data/equities'
    pmem = True if torch.cuda().is_available() else False
    stock_dataset = StockDataset(ticker_dir, index_col='Date')
    data_loader = DataLoader(stock_dataset, batch_size=4,
                             shuffle=True, pin_memory=pmem,
                             num_workers=4, collate_fn=StockBatch)

    for i, sample in enumerate(data_loader):
        logger.info(f'batch idx: {i} X size: {sample.X.size()}'
                    f'y size: {sample.y.size()}')

    logger.info('Finished Testing Stock Data Loader')


if __name__ == '__main__':
    main()
