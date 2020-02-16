import argparse
import time
import datetime as dt
import logging
from os import cpu_count, listdir
from os.path import isfile, join
from multiprocessing import Pool, current_process, get_logger
from itertools import product
import pandas as pd
import yfinance as yf
import s3fs
import numpy as np
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
        'scrty_ds', 'cpn_rt', 'close_pr', 'close_yld',
        'trans_dt', 'last_dt',
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

        # period count (2 year default) used for flattened columns
        self._pcnt = 8

        # column setup
        self._ins_cols = CBH.INS
        self._cfs_cols = CBH.CFS
        self._bs_cols = CBH.BS

        self._ins_chg_cols = [f'{c}_yoy_chg' for c in self._ins_cols]
        self._cfs_chg_cols = [f'{c}_yoy_chg' for c in self._cfs_cols]
        self._bs_chg_cols = [f'{c}_yoy_chg' for c in self._bs_cols]

        self._base_cols = self._ins_cols + self._cfs_cols + self._bs_cols
        self._chg_cols = self._ins_chg_cols + \
            self._cfs_chg_cols + self._bs_chg_cols
        self._dffin_cols = list(set(self._base_cols +
                                    self._chg_cols + self.SHARE_COLS))

        prng = range(self._pcnt)
        self._ev_adds = [f'{c}_0' for c in self.EV_ADDS]
        self._ev_subs = [f'{c}_0' for c in self.EV_SUBS]

        ev_outs = [c for c in self._dffin_cols if c not in self.SHARE_COLS]
        self._ev_cols = [f'{c}_{i}' for c in ev_outs for i in prng]
        self._out_cols = self.TX_COL_DESC + self._ev_cols

        # placeholders
        self._dftxs = None
        self._dflinks = None

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

        # s3 connection
        conn = s3fs.S3FileSystem(anon=False)

        # check financials exist
        if not conn.exists(financials_path):
            if self.verbosity > 0:
                msg = (
                    f'{ticker.upper()} has no financials '
                    f'at {financials_path}'
                )
                print(msg)
            return

        with conn.open(financials_path, 'rb') as f:
            dffin = pd.read_csv(f)

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
                f'{dffin.shape[0]} relevant financials found for {ticker} '
                f'from {financials_path}'
            )
            print(msg)

        # return None if zero relevant financials
        if dffin.shape[0] == 0:
            return

        return dffin

    def load_eqy_pxs(self, eqy_pxs_path):
        # get ticker from path
        loc = eqy_pxs_path.split('/')
        ticker = loc[-1].split('.csv')[0].upper()

        # s3 connection
        conn = s3fs.S3FileSystem(anon=False)

        # check price data exists
        if not conn.exists(eqy_pxs_path):
            return

        # load px data
        with conn.open(eqy_pxs_path, 'rb') as f:
            eqypxs = pd.read_csv(f)
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

    def fetch_eqy_pxs(self, tickers, secs=1, save_dir=None):
        # pulls daily historical equity prices from yahoo finance
        # prices pulled over max history available enabling raw close calc
        # data stored in descending order with recent data at top
        # raw stock price calculated from cumulative product of splits
        hit, missed = [], []
        total = 0
        for ticker in tickers:
            tick = yf.Ticker(ticker)
            ydata = tick.history(period='max',
                                 auto_adjust=False)
            if ydata.shape[0] > 0:
                ydata = ydata.sort_index(ascending=False)
                ydata['AdjSplits'] = ydata['Stock Splits']
                ydata.loc[ydata.AdjSplits == 0, 'AdjSplits'] = 1.0
                ydata['CumSplits'] = ydata.AdjSplits.cumprod()
                ydata['RawClose'] = ydata['Adj Close']*ydata.CumSplits
                total += ydata.shape[0]
                hit.append(ticker)

                if save_dir is not None:
                    ydata.to_csv(f'{save_dir}/{ticker.upper()}.csv',
                                 index=True)
            else:
                missed.append(tick)

            if self.verbosity > 0:
                complete = (len(hit)+len(missed))/len(tickers)*100
                lgr = get_logger()
                lgr.info(f'{ticker} | hits: {len(hit):,} ({complete:.0f})%')

            # don't overwhelm yhoo and get yourself throttled
            time.sleep(secs)

        return hit, missed

    def build_txs(self, ticker, eqy_pxs_dir=None,
                  financials_dir=None, save_dir=None):
        # get id for logging
        p = current_process()._identity
        jid = 0 if len(p) == 0 else p[0]

        # load relevant data
        dffin = self.load_financials(ticker, financials_dir)
        if dffin is None:
            return jid, None

        # group financials by trailing 2 year period
        dlt = pd.offsets.DateOffset(years=1, months=11)
        d = dffin.earnings_release_date
        masks = [(d > i-dlt) & (d <= i) for i in d]
        data = [dffin[mask] for mask in masks]
        data = pd.concat(data)

        midxs = [self._build_mi(t, idx, masks, d) for idx, t in enumerate(d)]
        midxs = [i for g in midxs for i in g]
        mi = pd.MultiIndex.from_tuples(midxs, names=['last_dt', 'per_dt'])

        dffin = pd.DataFrame(data.values,
                             index=mi,
                             columns=data.columns.values)
        dffin = dffin.reset_index()
        dffin = dffin.sort_values(['last_dt', 'per_dt'],
                                  ascending=[True, True])
        dffin = dffin.groupby('last_dt').filter(lambda g: len(g) == 8)

        # check whether there are relevant financials
        if dffin.shape[0] == 0:
            return jid, None

        # flatten financials into trailing 2 year period
        groups = dffin.groupby('last_dt')[self._dffin_cols]
        dffin_flat = groups.apply(self._flattener).droplevel(level=1)
        dffin_flat = dffin_flat.reset_index().sort_values(by=['last_dt'])

        # pull equity prices
        df_eqypxs = self.load_eqy_pxs(f'{eqy_pxs_dir}/{ticker}.csv')
        if df_eqypxs is None or df_eqypxs.shape[0] == 0:
            return jid, None

        # filter prices for relevant transaction dates
        dftxs = self.get_txs_by_ticker(ticker)
        df_tx_in = pd.merge(left=dftxs.set_index('trans_dt'),
                            right=df_eqypxs.set_index('Date'),
                            how='inner', left_index=True,
                            right_index=True)
        df_tx_in = df_tx_in.drop_duplicates()
        df_tx_in.index.name = 'trans_dt'
        df_tx_in = df_tx_in.reset_index().sort_values(by=['trans_dt'],
                                                      ascending=True)

        # check whether there are relevant rxs
        if df_tx_in.shape[0] == 0:
            return jid, None

        # merge with financials
        df_tx_in = pd.merge_asof(df_tx_in, dffin_flat,
                                 left_on=['trans_dt'],
                                 right_on=['last_dt'],
                                 direction='backward',
                                 tolerance=pd.Timedelta('90d')
                                 ).dropna()

        # ev normalization
        # historical ev based on raw close (unadjusted splits price)
        base = df_tx_in[self._ev_adds].sum(axis=1) - \
            df_tx_in[self._ev_subs].sum(axis=1)
        mkt_cap = df_tx_in.RawClose * df_tx_in.avgdilutedsharesoutstanding_0
        df_tx_in[self._ev_cols] = df_tx_in[self._ev_cols].div(base + mkt_cap,
                                                              axis=0)

        # pull relevant cols and clean up
        df_tx_in = df_tx_in[self._out_cols].dropna()

        if save_dir is not None:
            txcnt = df_tx_in.shape[0]
            if txcnt > 0:
                # locally saved to data/bonds/transactions/ticker.csv
                tx_out_path = f'{save_dir}/{ticker.upper()}.csv'
                df_tx_in.to_csv(tx_out_path, index=False)

        return jid, df_tx_in

    def _build_mi(self, ts, idx, masks, dates):
        return product([np.datetime64(ts)], dates[masks[idx]].values)

    def _flattener(self, g):
        cols = g.columns.values
        flat_cols = [f'{c}_{i}' for c in cols for i in range(g.shape[0])]
        data = np.concatenate([g[c].values for c in cols])
        return pd.DataFrame(data[None, :], columns=flat_cols, index=[g.name])

    def old_build_txs(self, ticker, eqy_pxs_dir=None,
                      financials_dir=None, save_dir=None):
        # load relevant data
        dffin = self.load_financials(ticker, financials_dir)
        if dffin is None:
            if self.verbosity > 0:
                print(f'skipping {ticker} -- no financials found')
            return

        df_eqypxs = self.load_eqy_pxs(f'{eqy_pxs_dir}/{ticker}.csv')
        if df_eqypxs is None:
            if self.verbosity > 0:
                print(f'skipping {ticker} -- no equity prices found')
            return

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

        # base ev input calcs
        dffin['ev_inputs'] = dffin[self.EV_ADDS].sum(axis=1) - \
            dffin[self.EV_SUBS].sum(axis=1)
        dffin = dffin.reset_index()

        # track state for logging
        valid, prior_valid, completed = 0, 0, 0
        total = df_tx_in.shape[0]
        prior_dt = dt.datetime.now()
        jid = current_process()._identity[0]
        for tdt, tx in df_tx_in.iterrows():
            if self.verbosity > 0:
                if (completed % (1000/self.verbosity) == 0) & (completed > 0):
                    curr_dt = dt.datetime.now()
                    dt_str = curr_dt.strftime('%H:%M:%S')
                    dt_diff = curr_dt-prior_dt
                    prior_dt = curr_dt
                    if dt_diff.seconds > 0:
                        pace = (valid-prior_valid)/dt_diff.seconds*60
                        eta = (total-completed)/pace*valid/completed
                    else:
                        pace = 0
                        eta = 0
                    prior_valid = valid

                    prefix = f'[{dt_str} ~ jid {jid} | {ticker}] '
                    comp_str = (
                        f'complete => {completed/total*100:,.1f}% '
                        f'({completed:,}/{total:,}) | '
                    )
                    val_str = (
                        f'valid => {valid/completed*100:,.1f}% '
                        f'({valid:,}/{completed:,}) | '
                    )
                    pace_str = (
                        f'valid/min => {pace:,.1f} | '
                        f'ETA => {eta:,.1f} min'
                    )
                    print(prefix, comp_str, val_str, pace_str)

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

                # state update for logging
                valid += 1

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
                print(f'{txcnt} {ticker} {txstr} saved to {postfix}')
        return df_flat


def progress_msg(ticker, st, jid, tx_cnt):
    et = dt.datetime.now()
    curr_dtime_diff = et-st

    prefix = f'[jid {jid} | {ticker}] '
    comp_str = f'txs => {tx_cnt:,} | '

    # only display relevant pace info
    if curr_dtime_diff.seconds <= 0:
        pace_str = ''
    else:
        curr_tx_pace = tx_cnt/curr_dtime_diff.seconds*60
        pace_str = f'txs/min => {curr_tx_pace:,.0f} '

    return ' '.join([prefix, comp_str, pace_str])


def spawn_cleaner(ticker, kwargs):
    st = dt.datetime.now()
    cbc = CBCleaner(links_path=kwargs['links_path'],
                    bond_txs_path=kwargs['bond_txs_path'],
                    verbosity=kwargs['verbosity'])
    jid, df = cbc.build_txs(ticker, eqy_pxs_dir=kwargs['eqy_pxs_dir'],
                            financials_dir=kwargs['financials_dir'],
                            save_dir=kwargs['save_dir'])
    lgr = get_logger()
    tx_cnt = df.shape[0] if df is not None else 0
    lgr.info(progress_msg(ticker, st, jid, tx_cnt))


EXCLUDES = [
    'ABCD', 'ABS', 'ABX', 'ACL', 'ACV', 'ADPT', 'AEGR', 'AEPI',
    'AFFX', 'AER', 'AFSI', 'AIQ', 'AL', 'ALJ', 'ALTR', 'AM',
    'AMMD', 'AMRE', 'AMLN', 'AMRI', 'AONE', 'ANR', 'ANV', 'APFC',
    'APL', 'APC', 'ARDX', 'ARG', 'ASCA', 'ASCMA', 'ARRY', 'AT',
    'ARII', 'ATPG', 'AVTR', 'AVIV', 'AUDC', 'AUXL', 'AWH', 'BABA',
    'BCE', 'BBEP', 'BEAV', 'BIDU', 'BBG', 'BCR', 'BJS', 'BIOA',
    'BEL', 'BKW', 'BLC', 'BMO', 'BMR', 'BG', 'BNS', 'BMS', 'BP',
    'BONT', 'BRCM', 'BPL', 'BRK', 'BRCD', 'EEQ', 'EQ', 'DLPH',
    'CSIQ', 'FSL', 'ENH', 'BRSS', 'EFII', 'DRC', 'CVC', 'DYN',
    'EAC', 'EXAM', 'ENOC', 'ENP', 'GENZ', 'GEOY', 'FTK', 'EXL',
    'EXXI', 'CBST', 'CIT', 'CSR', 'CKEC', 'CA', 'GGS', 'CYN',
    'CVRR', 'BSFT', 'CWEI', 'CSTR', 'CSWC', 'CNF', 'CNHI',
    'FCEA', 'FLR', 'FFC', 'CLWR', 'CFBI', 'CWTR', 'CCIX', 'CNK',
    'CNL', 'CRDN', 'FLY', 'CLD', 'DRYS', 'DT', 'CRTX', 'GAS',
    'DNDN', 'CHRS', 'CNQR', 'CRWN', 'END', 'BTU', 'CHTR', 'FIG',
    'FISH', 'BWP', 'ACET', 'EPB', 'APU', 'HTCH', 'ITMN', 'GOK',
    'DVR', 'CMLS', 'IM', 'IPCC', 'HE', 'DLLR', 'HOT', 'HS',
    'EROC', 'CMRE', 'HLT', 'HMA', 'HELI', 'ENGY', 'ESL', 'EVER',
    'DOW', 'DOX', 'CBE', 'CPHD', 'CPNO', 'HGSI', 'HIW', 'INSP',
    'DRIV', 'CPX', 'CQB', 'JONE', 'JRCC', 'HAR', 'CENX', 'CEPH',
    'IRC', 'GR', 'GRM', 'KEYW', 'KRN', 'KGC', 'IX', 'JAG',
    'JAKK', 'ITC', 'ITP', 'EDE', 'EDR', 'HTLF', 'HTWR', 'LMIA', 'EPL', 'LBTYA',
    'I', 'LCC', 'MAIN', 'JD', 'JDAS', 'MANT', 'LVLT', 'IDC', 'IDTI', 'LNCR',
    'LNKD', 'MCP', 'MDAS', 'FNSR', 'LVS', 'EQY', 'MDSO', 'FRZ', 'MHGC',
    'INVN', 'IGT', 'CIE', 'MPG', 'MPO', 'LEXG', 'LGF', 'MHS', 'DFT', 'MRD',
    'MRGE', 'MDP', 'MRH', 'MIG', 'NHP', 'KOG', 'N', 'MIR', 'MDSN', 'MAST',
    'NSR', 'HCCH', 'HCM', 'NAVG', 'CALD', 'CALL', 'NTI', 'NTK', 'NFP', 'NIHD',
    'NKA', 'NWK', 'DB', 'NOR', 'NPBC', 'NRX', 'KWK', 'DBD', 'DBRN', 'NGLS',
    'NH', 'CME', 'DCT', 'NVS', 'NMR', 'MELI', 'MENT', 'OCR', 'OEC', 'LAYN',
    'NWS', 'NXPI', 'PKD', 'NEWS', 'CTV', 'NSM', 'CXP', 'PRFT', 'PWAV', 'P',
    'PLL', 'PX', 'PRI', 'PTP', 'PTRY', 'PTV', 'REV', 'MF', 'REXX', 'RFMD',
    'OMED', 'PVR', 'PRSC', 'ONCS', 'ONE', 'RGC', 'RRMS', 'SCTY', 'PCL',
    'RRR', 'MON', 'RICE', 'SD', 'PCP', 'SDC', 'PD', 'QLTY', 'QRE', 'SFG',
    'SFLY', 'SFUN', 'SGY', 'MOTR', 'SHLM', 'SIX', 'SQNM', 'SUSS', 'SVM',
    'PBG', 'OREX', 'RY', 'SEP', 'SVU', 'RA', 'TDW', 'RYL', 'RSPP', 'RTI',
    'PMDX', 'TNB', 'TRAK', 'TRNX', 'TRP', 'PPC', 'STR', 'RAH', 'TRS', 'RAS',
    'RATE', 'TWTC', 'STRI', 'SALM', 'SARA', 'TRW', 'TXI', 'TRH', 'TRK',
    'TRLA', 'SUG', 'TSRO', 'ROC', 'UDRL', 'PBR', 'TAL', 'TAM', 'SUNH',
    'SUSQ', 'TUC', 'TOL', 'TOT', 'TSS', 'TSYS', 'PBY', 'ROSE', 'OXM', 'VQ',
    'VTSS', 'TD', 'TK', 'USAS', 'SCHS', 'TPCG', 'WCRX', 'VR', 'TLP', 'USG',
    'VRS', 'SPI', 'PSSI', 'TYC', 'UIL', 'WNR', 'WOLF', 'XIDE', 'SPLS',
    'SPNC', 'PPO', 'PPS', 'ZZ', 'TZOO', 'UBS', 'WB', 'WBMD', 'VBLT', 'XL',
    'WPZ', 'VC', 'VCI', 'UTIW', 'SPTN', 'WWAV', 'WRES', 'WSTC', 'XNPT',
    'VTG', 'STEI', 'VIAS', 'VLP', 'WGL', 'WH', 'XTO', 'XXIA', 'VMEM',
    'VNR', 'VOLC', 'VPHM', 'XRM', 'XCO', 'MWE', 'SLXP', 'PGEM', 'URS',
    'GST', 'GTIV', 'GM', 'GMXR', 'CIK0001645494', 'LZ', 'FDO', 'FES',
    'KFN', 'LLTC', 'SIAL', 'SONO', 'SNDK', 'SPC', 'LXK', 'MJN', 'REN', 'SYA',
    'LLL', 'RDC', 'SWY', 'TE', 'NG', 'RESI', 'TWGP', 'TWC', 'TEP', 'SNX',
    'WIN', 'MA', 'TERP', 'WMG'
]

if __name__ == '__main__':
    # setup cli arg parser
    base_s3 = 's3://elm-credit'
    parser = argparse.ArgumentParser(description='Transaction Preprocessing')
    parser.add_argument('--links_path',
                        default=f'data/ciks/bonds_to_equities_link.csv',
                        help='ticker cusip links path')
    parser.add_argument('--bond_txs_path',
                        default=f'data/bonds/clean_bond_close_pxs.csv',
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
    parser.add_argument('--limit',
                        type=int,
                        help='ticker processing limit count')
    args = parser.parse_args()

    if args.pool_size is not None and args.pool_size < 1:
        msg = f'poolsize must be an integer >= 1, got {args.pool_size}'
        raise ValueError(msg)

    # initialize logger
    logger = get_logger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # s3 connection
    conn = s3fs.S3FileSystem(anon=False)

    # get list of tickers
    ufp = args.financials_dir.split('s3://')[-1]
    if not conn.exists(ufp):
        logger.error(f'ticker list not found at {ufp}')
        exit()

    universe = conn.ls(ufp)
    if len(universe) > 1:
        universe = [f.split('/')[-1].split('.csv')[0] for f in universe]
        universe = universe[1:]
    else:
        raise ValueError('no tickers found at {args.financials_dir}')

    if args.save_dir.startswith('s3://'):
        # check aws
        pfp = args.save_dir.split('s3://')[-1]
        if not conn.exists(pfp):
            logger.error(f'prior transactions list not found at {pfp}')
            exit()
        priors = conn.ls(pfp)
    else:
        # check locally
        sd = args.save_dir
        priors = [f for f in listdir(sd) if isfile(join(sd, f))]

    if len(priors) > 1:
        priors = [f.split('/')[-1].split('.csv')[0] for f in priors]
        priors = priors[1:]

    ffp = args.financials_dir.split('s3://')[-1]
    if not conn.exists(ffp):
        logger.error(f'financials list not found at {ffp}')
        exit()

    fins = conn.ls(ffp)
    if len(fins) > 1:
        fins = [f.split('/')[-1].split('.csv')[0] for f in fins]
        fins = fins[1:]
    else:
        raise ValueError('no financials found at {args.financials_dir}')

    # clear fs cache to ensure multiprocessing instances created
    conn.clear_instance_cache()

    # filter prior and no financials from tickers
    # filter prior misses (no relevant financials or equity prices)
    # limit processing based on cli input
    ts_miss = priors + EXCLUDES
    ts = [t for t in universe if t not in ts_miss]
    ts = [t for t in ts if t in fins]
    remain = len(ts)
    if args.limit:
        ts = ts[:args.limit]
    logger.info(f'processing {len(ts)} of {remain} remaining tickers')

    date_str = dt.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    ticker_list_path = f'data/ticker_pipeline_{date_str}.csv'
    pd.DataFrame({'ticker': ts}).to_csv(ticker_list_path, index=False)
    logger.info(f'working tickers saved to {ticker_list_path}')

    # single builder unless multiple tickers and pool or pool_size arg supplied
    params = vars(args)
    if len(ts) < 0:
        msg = f'no tickers found for cleaning'
        raise ValueError(msg)
    elif (args.pool is False) & (args.pool_size is None):
        logger.info('spawning cleaners on single process')
        for t in ts:
            spawn_cleaner(t, params)
    elif len(ts) == 1:
        logger.info('spawning cleaners on single process')
        spawn_cleaner(ts[0], params)
    else:
        # cap builders by ticker length
        bcnt = min(len(ts), cpu_count())
        if args.pool_size is not None:
            bcnt = min(len(ts), args.pool_size)

        logger.info(f'spawning cleaners on {bcnt} processes')
        params = [(t, params) for t in ts]
        with Pool(processes=bcnt) as p:
            p.starmap(spawn_cleaner, params)
