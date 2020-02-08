import argparse
import time
import datetime as dt
from os import cpu_count
from os.path import join
from multiprocessing import Pool
import pandas as pd
import yfinance as yf
import s3fs
from CalcBenchHandler import CalcBenchHandler as CBH


class CBCleaner(CBH):
    FS_COLS = CBH.INS+CBH.CFS+CBH.BS
    SHARE_COLS = [
        'sharesoutstandingendofperiod',
        'avgsharesoutstandingbasic',
        'avgdilutedsharesoutstanding',
        'stockrepurchasedduringperiodshares',
        'commonstockdividendspershare'
    ]
    DEBT_COLS = [
        'currentlongtermdebt', 'longtermdebt',
        'totaldebt', 'lineofcreditfacilityamountoutstanding',
        'secureddebt', 'seniornotes', 'subordinateddebt',
        'convertibledebt', 'termloan', 'mortgagedebt',
        'unsecureddebt', 'mediumtermnotes',
        'trustpreferredsecurities'
    ]
    EV_ADDS = [
        'currentlongtermdebt', 'longtermdebt',
        'restrictedcashandinvestmentscurrent',
        'trustpreferredsecurities'
    ]
    EV_SUBS = [
        'cash', 'availableforsalesecurities',
        'totalinvestments'
    ]
    TX_COL_DESC = [
        'mtrty_dt', 'bond_sym_id',
        'company_symbol', 'issuer_nm', 'debt_type_cd',
        'scrty_ds', 'cpn_rt', 'close_pr', 'close_yld'
    ]

    def __init__(self, links_path=None, bond_txs_path=None,
                 verbosity=0):
        super().__init__()

        # check args
        if verbosity in range(3):
            self.verbosity = int(verbosity)
        else:
            emsg = f'verbosity must be in {range(3)}; got {verbosity}'
            raise ValueError(emsg)

        # data dirs
        self.links_path = links_path
        self.bond_txs_path = bond_txs_path

        # column setup
        self._ins_cols = [c for c in CBH.INS if c not in self.SHARE_COLS]
        self._cfs_cols = [c for c in CBH.CFS if c not in self.SHARE_COLS]
        self._bs_cols = [c for c in CBH.BS if c not in self.SHARE_COLS]

        self._ins_chg_cols = [f'{c}_yoy_chg' for c in self._ins_cols]
        self._cfs_chg_cols = [f'{c}_yoy_chg' for c in self._cfs_cols]
        self._bs_chg_cols = [f'{c}_yoy_chg' for c in self._bs_cols]

        self._base_cols = self._ins_cols + self._cfs_cols + self._bs_cols
        self._chg_cols = self._ins_chg_cols + \
            self._cfs_chg_cols + self._bs_chg_cols
        self._out_cols = self._base_cols + self._chg_cols

        # period count (2 year default) used for flattened columns
        self._pcnt = 8
        prng = range(self._pcnt)
        self._cols_flat = [f'{c}_{i}' for c in self._out_cols for i in prng]

        # placeholders
        self._dftxs = None
        self._dflinks = None

        # display settings
        self.print_settings()

    def print_settings(self):
        prefix = '\t'
        print('\nCleaner Settings:')
        print(prefix, f'verbosity: {self.verbosity}')
        print()

    def load_links(self):
        dflink = pd.read_csv(self.links_path)

        # drop duplicate symbol/equity_cusip records
        dfdupes = dflink.groupby(['SYMBOL', 'EQUITY_CUSIP']).count()
        sym_counts = dfdupes.index.get_level_values(0).value_counts()
        sym_dupes = sym_counts[sym_counts > 1]
        dupes = dfdupes.reset_index()
        mask = (~dupes.SYMBOL.isin(sym_dupes.index.values))
        ser_eqy_cusip = dupes[mask].EQUITY_CUSIP
        self._dflinks = dflink[dflink.EQUITY_CUSIP.isin(ser_eqy_cusip)]

        if self.verbosity == 2:
            msg = (
                f'{self._dflinks.shape[0]} links loaded from '
                f'{self.links_path}'
            )
            print(msg)

    def load_bond_txs(self):
        self._dftxs = pd.read_csv(self.bond_txs_path)

        # bond prices must have a transaction date
        self._dftxs.trans_dt = pd.to_datetime(self._dftxs.trans_dt,
                                              errors='coerce')
        self._dftxs = self._dftxs.dropna(subset=['trans_dt'])

        if self.verbosity == 2:
            msg = (
                f'{self._dftxs.shape[0]} bond transactions '
                f'loaded from {self.bond_txs_path}'
            )
            print(msg)

    def load_financials(self, ticker, financials_dir):
        # get list of tickers with available financials
        financials_path = join(financials_dir, f'{ticker}.csv')
        dffin = pd.read_csv(financials_path)

        # clean earnings_release_date
        dffin.earnings_release_date = pd.to_datetime(
            dffin.earnings_release_date,
            errors='coerce')
        dffin = dffin.dropna(subset=['earnings_release_date'])
        dffin = dffin.sort_values(by='earnings_release_date', ascending=False)

        # fill na income statement fields with with 0
        dffin = dffin.fillna(value={k: 0 for k in self.FS_COLS})

        # compute yoy chg
        dffin[self._chg_cols] = dffin[self._base_cols] - \
            dffin[self._base_cols].shift(4)
        dffin = dffin.dropna()

        # check all fields have values
        assert dffin.isna().sum().sum() == 0, f'{ticker} has na fields!'

        if self.verbosity == 2:
            msg = (
                f'{dffin.shape[0]} financials found for {ticker} '
                f'from {financials_path}'
            )
            print(msg)

        return dffin

    def load_eqy_pxs(self, eqy_pxs_path):
        # get ticker from path
        loc = eqy_pxs_path.split('/')
        ticker = loc[-1].split('.csv')[0].upper()

        # load px data
        eqypxs = pd.read_csv(eqy_pxs_path)
        eqypxs = eqypxs.rename(columns={'Adj Close': 'AdjClose'})

        if (self.verbosity > 1) & (eqypxs.shape[0] <= 0):
            msg = (
                f'{ticker} does not have '
                f'any equity price data available '
                f'at {eqy_pxs_path}'
            )
            print(msg)

        return eqypxs

    def get_txs_by_ticker(self, ticker):
        # load links if not done already
        if self._dflinks is None:
            self.load_links()

        # load txs if not done already
        if self._dftxs is None:
            self.load_bond_txs()

        # get links to bond cusips for ticker
        links = self._dflinks[self._dflinks.SYMBOL == ticker]

        # get txs for cusips
        txs = self._dftxs[self._dftxs.cusip_id.isin(links.cusip_id)]
        txs = txs.sort_values(by='trans_dt', ascending=False)

        return txs

    def fetch_eqy_pxs(self, tickers, sd='2009-01-01', ed='2020-01-28',
                      secs=1, save_dir=None):
        # pulls daily historical equity prices from yahoo finance
        hit, missed = [], []
        total = 0
        ycols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        for ticker in tickers:
            tick = yf.Ticker(ticker)
            ydata = tick.history(start=sd, end=ed)
            if ydata.shape[0] > 0:
                ydata['Adj Close'] = ydata.Close
                ydata = ydata[ycols].sort_index(ascending=False)
                total += ydata.shape[0]
                hit.append(ticker)

                if save_dir is not None:
                    ydata.to_csv(f'{save_dir}/{ticker.upper()}.csv',
                                 index=True)

                if self.verbosity == 2:
                    msg = (
                        f'{ydata.shape[0]} {ticker.upper()} '
                        f'equity price data saved to '
                        f'{save_dir}/{ticker.upper()}.csv'
                    )
                    print(msg)
            else:
                missed.append(tick)

                if self.verbosity == 2:
                    msg = (
                        f'no equity price data found for {ticker.upper()} '
                    )
                    print(msg)

            # don't overwhelm yhoo and get yourself throttled
            time.sleep(secs)

        if self.verbosity > 0:
            print('\nEquity Data Pull Summary:')
            prefix = '\t'
            print(prefix, f'price count: {total}')
            print(prefix, f'ticker hit count: {len(hit)}')
            print(prefix, f'ticker miss count: {len(missed)}')
            print()

        return hit, missed

    def build_txs(self, ticker, eqy_pxs_dir=None,
                  financials_dir=None, save_dir=None):
        # load relevant data
        dffin = self.load_financials(ticker, financials_dir)
        df_eqypxs = self.load_eqy_pxs(f'{eqy_pxs_dir}/{ticker}.csv')
        dftxs = self.get_txs_by_ticker(ticker)

        # filter prices for relevant transaction dates
        df_tx_in = pd.merge(left=dftxs.set_index('trans_dt'),
                            right=df_eqypxs.set_index('Date'),
                            how='inner', left_index=True,
                            right_index=True)
        df_tx_in.index.name = 'trans_dt'
        df_tx_in = df_tx_in.drop_duplicates()
        df_tx_in.index.name = 'trans_dt'

        mi = pd.MultiIndex.from_tuples(zip(df_tx_in.cusip_id, df_tx_in.index),
                                       names=['cusip_id', 'trans_dt'])
        df_flat = pd.DataFrame(columns=self.TX_COL_DESC+self._cols_flat,
                               index=mi)
        df_flat = df_flat.sort_index()
        df_flat[self.TX_COL_DESC] = df_tx_in[self.TX_COL_DESC].values

        completed = 0
        total = df_tx_in.shape[0]

        # base ev input calcs
        dffin['ev_inputs'] = dffin[self.EV_ADDS].sum(axis=1) - \
            dffin[self.EV_SUBS].sum(axis=1)
        dffin = dffin.reset_index()

        for tdt, tx in df_tx_in.iterrows():
            if self.verbosity > 0:
                if completed % (1000/self.verbosity) == 0:
                    print('\t', f'{ticker}: {completed}/{total} txs cleaned')

            # find most recent period
            df_2ltm = dffin[tdt >= dffin.earnings_release_date].head(8)
            m2 = tdt-df_2ltm.earnings_release_date <= dt.timedelta(90)
            if (m2.sum() > 0) & (df_2ltm.shape[0] == self._pcnt):
                per = df_2ltm.iloc[0]
                mkt_cap = tx.AdjClose*per.avgdilutedsharesoutstanding
                df_2ltm[self._out_cols] /= per.ev_inputs+mkt_cap

                # stack rows into single column
                outs = df_2ltm[self._out_cols].values.ravel()
                df_flat.loc[(tx.cusip_id, tdt), self._cols_flat] = outs

            completed += 1

        # save flattened transactions
        df_flat = df_flat.dropna()

        if save_dir is not None:
            txcnt = df_flat.shape[0]
            if txcnt > 0:
                # locally saved to data/bonds/transactions/ticker.csv
                tx_out_path = f'{save_dir}/{ticker.upper()}.csv'
                df_flat.to_csv(tx_out_path, index=True)

            if self.verbosity > 0:
                txstr = 'transaction' if txcnt == 1 else 'transactions'
                postfix = f'{tx_out_path}' if txcnt > 0 else ''
                print(f'{txcnt} {ticker} {txstr} saved {postfix} to ')
        return df_flat


def spawn_cleaner(ticker, kwargs):
    cbc = CBCleaner(links_path=kwargs['links_path'],
                    bond_txs_path=kwargs['bond_txs_path'],
                    verbosity=kwargs['verbosity'])
    cbc.build_txs(ticker, eqy_pxs_dir=kwargs['eqy_pxs_dir'],
                  financials_dir=kwargs['financials_dir'],
                  save_dir=kwargs['save_dir'])


if __name__ == '__main__':
    # setup cli arg parser
    base_s3 = 's3://elm-credit'
    parser = argparse.ArgumentParser(description='Transaction Preprocessing')
    parser.add_argument('--links_path',
                        default=f'elm-credit/ciks/bonds_to_equities_link.csv',
                        help='ticker cusip links path')
    parser.add_argument('--bond_txs_path',
                        default=f'elm-credit/bonds/clean_bond_close_pxs.csv',
                        help='bond transactions path')
    parser.add_argument('--eqy_pxs_dir',
                        default=f'{base_s3}/equities',
                        help='equity prices directory')
    parser.add_argument('--financials_dir',
                        default=f'{base_s3}/financials',
                        help='financials directory')
    parser.add_argument('--save_dir',
                        default=f'{base_s3}/bonds/transactions',
                        help='cleaned transaction save directory')
    parser.add_argument('--pool',
                        action='store_true',
                        help='multiprocessing with max cpus')
    parser.add_argument('--pool_size',
                        type=int,
                        help='multiprocessing pool size (overrides --pool)')
    parser.add_argument('--verbosity',
                        choices=[0, 1, 2], type=int, default=0,
                        help='verbosity level (0=off, 1=mild, 2=high)')
    args = parser.parse_args()

    if args.pool_size is not None and args.pool_size < 1:
        msg = f'poolsize must be an integer >= 1, got {args.pool_size}'
        raise ValueError(msg)

    # get list of tickers
    fs = s3fs.S3FileSystem(anon=False)
    ufp = args.financials_dir.split('s3://')[-1]
    universe = fs.ls(ufp)
    if len(universe) > 1:
        universe = [f.split('.csv')[0] for f in universe]
        universe = universe[1:]
    else:
        raise ValueError('no tickers found at {args.financials_dir}')

    pfp = args.save_dir.split('s3://')[-1]
    priors = fs.ls(pfp)
    if len(priors) > 1:
        priors = [f.split('.csv')[0] for f in priors]
        priors = priors[1:]
    else:
        raise ValueError('no prior transactions found at {args.save_dir}')

    ts = [t for t in universe if t not in priors]
    print(f'remaining tickers: {len(ts)}')

    # single builder unless multiple tickers and pool or pool_size arg supplied
    if len(ts) < 0:
        msg = f'no tickers found for cleaning'
        raise ValueError(msg)
    elif (args.pool is False) & (args.pool_size is None):
        print('spawning cleaners on single process')
        for t in ts:
            spawn_cleaner(t, vars(args))
    elif len(ts) == 1:
        print('spawning cleaners on single process')
        spawn_cleaner(ts[0], vars(args))
    else:
        # cap builders by ticker length
        bcnt = min(len(ts), cpu_count())
        if args.pool_size is not None:
            bcnt = min(len(ts), args.pool_size)

        print(f'spawning cleaners on {bcnt} processes')
        params = [(t, vars(args)) for t in ts]
        with Pool(processes=bcnt) as p:
            p.starmap(spawn_cleaner, params)
