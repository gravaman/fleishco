import numpy as np
import torch
from torch.utils.data import Dataset
from db.db_query import (
    get_corptx_ids,
    get_credit_data,
    get_fin_cols
)


class CreditDataset(Dataset):
    """Credit Specific Dataset"""

    # based on AAPL sample data (update as needed)
    EX_COLS_AAPL = [
        'assetimpairment', 'restructuring',
        'depreciationandamortizationexpense',
        'interestexpense', 'capitalassetsales',
        'other_opex', 'other_addbacks', 'dividends'
    ]
    EX_COLS_CHTR = [
        'sgaexpense', 'researchanddevelopment',
        'shortterminvestments', 'longterminvestments',
        'sharebasedcompensation', 'acquisitiondivestitures',
        'other_investments', 'assetimpairment',
        'restructuring', 'capitalassetsales', 'other_addbacks',
        'dividends'
    ]
    EX_COLS_MSFT = [
        'depreciationandamortizationexpense', 'interestexpense',
        'capitalassetsales', 'acquisitiondivestitures',
        'other_opex', 'other_addbacks'
    ]
    EX_COLS_IBM = [
        'depreciationandamortizationexpense', 'interestexpense',
        'capitalassetsales', 'acquisitiondivestitures',
        'other_opex', 'other_addbacks', 'longterminvestments'
    ]
    EX_COLS_QCOM = [
        'depreciationandamortizationexpense', 'acquisitiondivestitures',
        'other_addbacks', 'dividends', 'capitalassetsales'
    ]

    def __init__(self, tickers, T=8, limit=None, standardize=False,
                 exclude_cols=[], txids=None, standard_stats=None):
        """
        Initializes credit dataset

        params:
        tickers (list): credit tickers mapping to corp_tx table
        T (int): financial time series length
        limit (int): max number of txs in dataset
        standardize (bool): data standardization indicator
        standard_stats 2x((T,D_fin),(1,D_ctx) np): standardization stats
            where the first tuple is for the financials and second is for
            the context
        """
        # save state
        self.tickers = tickers
        self.T = T
        self.release_window = T*90+10  # 90 day filing windows + 10 day buffer
        if txids is not None:
            self.txids = np.array(txids)
        else:
            self.txids = get_corptx_ids(tickers,
                                        release_window=self.release_window,
                                        release_count=T, limit=limit)
        self.exclude_cols = exclude_cols

        # conditionally set standardization stats and indicator
        self.standardize = standardize
        if standardize and standard_stats is None:
            # get stats for standardization if not provided
            self.standard_stats = self.get_stats()
        elif standard_stats is not None:
            # store standardization stats (not necessarily going to use)
            self.standard_stats = standard_stats
        else:
            self.standard_stats = None

    def get_stats(self):
        """
        Calculates mean and standard deviation for dataset.

        returns
        stats (CrediStats): mu and std for financials and context
        """
        fin_cols = get_fin_cols(int(self.txids[0]),
                                release_window=self.release_window,
                                limit=self.T,
                                exclude_cols=self.exclude_cols)

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

        _, cols = (fin_std == 0).nonzero()

        if len(cols) > 0:
            zero_cols = fin_cols[np.unique(cols)]
            print(f'fin std zero for following columns: {zero_cols}')

        stats = CreditStats((fin_mu, fin_std), (ctx_mu, ctx_std))
        return stats

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
        if self.standardize:
            fin, ctx = self.standard_stats.standardize(fin, ctx)

        # convert to tensors
        Xfin = torch.tensor(fin, dtype=torch.float)
        Xctx = torch.tensor(ctx[:, :-1], dtype=torch.float)
        Y = torch.tensor(ctx[:, -1], dtype=torch.float)

        return Xfin, Xctx, Y


class CreditStats:
    def __init__(self, fin_stats, ctx_stats):
        """
        Initializes credit stats for financials and context.

        params
        fin_stats ((T,D_fin) np, (T,D_fin) np): fin stats
        ctx_stats ((1,D_ctx) np, (1,D_ctx) np): ctx stats with y last col
        """
        self.fin_stats = Stats(*fin_stats)
        self.ctx_stats = Stats(*ctx_stats)

    @property
    def target(self):
        """
        Gets target stats from context

        return
        mu ((1,1) np): mean of context target
        std ((1,1) np): std of context target
        """
        return self.ctx_stats.target

    def standardize(self, fin_data, ctx_data):
        """
        Standardizes given input data

        params
        fin_data ((T, D_fin) np): financials input data
        ctx_data ((1, D_ctx) np): context input data

        returns
        standard_fin ((T, D_fin) np): standardized financials data
        standard_ctx ((1, D_ctx) np): standardized context data
        """
        standard_fin = self.fin_stats.standardize(fin_data)
        standard_ctx = self.ctx_stats.standardize(ctx_data)
        return standard_fin, standard_ctx

    def destandardize(self, standard_fin, standard_ctx):
        """
        Reverts standardization for given input data

        params
        standard_fin ((T, D_fin) np): standardized financials data
        standard_ctx ((1, D_ctx) np): standardized context data

        returns
        fin_data ((T, D_fin) np): financials input data
        ctx_data ((1, D_ctx) np): context input data
        """
        fin_data = self.fin_stats.destandardize(standard_fin)
        ctx_data = self.ctx_stats.destandardize(standard_ctx)
        return fin_data, ctx_data


class Stats:
    def __init__(self, mu, std):
        """
        Initializes stats

        params
        mu ((T, D_in) np): mean across time series and feats
        std ((T, D_in) np): std across time serires and feats
        """
        # sanity check inputs
        assert len(mu.shape) == 2, f'mu dims {len(mu.shape)} != 2'
        assert len(std.shape) == 2, f'std dims {len(mu.shape)} != 2'
        assert not (std == 0).any(), f'std contains zero'
        self.mu = mu
        self.std = std

    @property
    def values(self):
        """
        Exposes stats

        returns
        mu ((T,D_in) np): mean across time series and feats
        std ((T,D_in) np): std across time series and feats
        """
        return self.mu, self.std

    @values.setter
    def values(self, mu, std):
        # clean and sanity check args
        mu, std = np.array(mu), np.array(std)
        assert mu.shape == self.mu.shape, \
            f'mu shape {mu.shape} != self.mu shape {self.mu.shape}'
        assert std.shape == self.std.shape, \
            f'std shape {std.shape} != self.std shape {self.std.shape}'
        assert not (std == 0).any(), f'std contains zero'
        # update stats
        self.mu = mu
        self.std = std

    @property
    def target(self):
        """
        Exposes target stats

        returns
        mu ((T, 1) np arr): mean across time series
        std ((T, 1) np arr): std across time series
        """
        return self.mu[:, -1], self.std[:, -1]

    def standardize(self, data, is_target=False):
        """
        Standardizes given input data

        params
        data ((T, D_in) np): input data to standardize
        is_target (bool): target indicator

        returns
        standard ((T, D_in) np): standardized data
        """
        if is_target:
            mu, std = self.target
        else:
            mu, std = self.mu, self.std

        return (data-mu)/std

    def destandardize(self, data, is_target=False):
        """
        Reverts standardization of given input data.

        params
        data ((T, D_in) np): input data to standardize
        is_target (bool): target indicator

        returns
        raw ((T, D_in) np): raw input data
        """
        if is_target:
            mu, std = self.target
        else:
            mu, std = self.mu, self.std

        return data.cpu()*std+mu
