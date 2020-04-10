import numpy as np
import torch
from torch.utils.data import Dataset
from db.db_query import get_corptx_ids, get_credit_data


class CreditDataset(Dataset):
    """Credit Specific Dataset"""

    # based on AAPL sample data (update as needed)
    EX_COLS = [
        'assetimpairment', 'restructuring',
        'depreciationandamortizationexpense',
        'interestexpense', 'capitalassetsales',
        'other_opex', 'other_addbacks', 'dividends'
    ]

    def __init__(self, ticker, T=8, limit=None, standardize=True,
                 exclude_cols=[]):
        """
        Initializes credit dataset

        params:
        ticker (str): credit ticker mapping to corp_tx table
        T (int): financial time series length
        limit (int): max number of txs in dataset
        standardize (bool): data standardization indicator
        """
        # save state
        self.ticker = ticker
        self.T = T
        self.release_window = T*90+10  # 90 day filing windows + 10 day buffer
        self.standardize = standardize
        self.txids = get_corptx_ids(ticker,
                                    release_window=self.release_window,
                                    release_count=T, limit=limit)
        self.exclude_cols = exclude_cols

        # conditionally standardize
        self.stats = None
        if standardize:
            # calculate mu
            total_fin, total_ctx = get_credit_data(
                id=int(self.txids[0]),
                release_window=self.release_window,
                limit=self.T, exclude_cols=self.exclude_cols)
            N = 0
            for txid in self.txids[1:]:
                fin, ctx = get_credit_data(
                    id=int(txid),
                    release_window=self.release_window,
                    limit=self.T, exclude_cols=self.exclude_cols)
                total_fin = total_fin+fin
                total_ctx = total_ctx+ctx
                N += 1

            fin_mu = total_fin/N
            ctx_mu = total_ctx/N

            # calculate std
            total_fin = np.zeros_like(total_fin)
            total_ctx = np.zeros_like(total_ctx)
            for txid in self.txids:
                fin, ctx = get_credit_data(
                    id=int(txid),
                    release_window=self.release_window,
                    limit=self.T, exclude_cols=self.exclude_cols)
                total_fin += (fin-fin_mu)**2
                total_ctx += (ctx-ctx_mu)**2

            fin_std = (total_fin/N)**0.5
            ctx_std = (total_ctx/N)**0.5
            self.stats = ((fin_mu, fin_std), (ctx_mu, ctx_std))

    def __len__(self):
        """Returns dataset length"""
        return len(self.txids)

    def __getitem__(self, idx):
        """
        Pulls financial, corp_tx data from postgres

        params
        idx (int): corp_tx element to pull

        return
        X_fin (nd torch.tensor): financial time series
        X_ctx (1d torch.tensor): corp_tx data
        Y (scalar torch.tensor): target close_yld
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # pull data from db
        fin, ctx = get_credit_data(id=int(self.txids[idx]),
                                   release_window=self.release_window,
                                   limit=self.T,
                                   exclude_cols=self.exclude_cols)

        # conditionally standardize
        if self.stats is not None:
            (fin_mu, fin_std), (ctx_mu, ctx_std) = self.stats
            fin = (fin-fin_mu)/fin_std
            ctx = (ctx-ctx_mu)/ctx_std

        # convert to tensors
        Xfin = torch.tensor(fin, dtype=torch.float).unsqueeze(0)
        Xctx = torch.tensor(ctx[:, :-1], dtype=torch.float).unsqueeze(0)
        Y = torch.tensor(ctx[:, -1], dtype=torch.float).unsqueeze(0)

        return Xfin, Xctx, Y
