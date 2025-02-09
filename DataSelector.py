from os import listdir
from os.path import join, isfile
from datetime import datetime
from uuid import uuid4
import pandas as pd
import numpy as np
import s3fs


class DataSelector:
    S3_PREFIX = 's3://'
    S3_BASE = 'elm-credit'
    LOCAL_BASE = 'data'
    BONDS_TXS_DIR = 'bonds/transactions'  # cleaned bond txs
    CIKS_META_PATH = 'ciks/companies.csv'  # ticker/cik meta info (naics, sic)
    CIK_CUSIP_LINK_PATH = 'ciks/CIK_CUSIP.csv'  # cik cusip linker
    BOND_MASTER_PATH = 'bonds/master_file.csv'  # issuer meta by cusip
    TS_OUTPUT_DIR = 'bonds/ticker_lists'  # output dir for ticker meta storage
    DATA_OUTPUT_DIR = 'bonds/combined'  # output dir for consolidated datasets
    RATES_BAML_DIR = 'rates/baml'  # ytm by duration and credit rating
    RATES_UST_DIR = 'rates/ust'  # ytm by duration for us treasuries
    RATES_OUTPUT_DIR = 'rates/combined'  # output dir for consolidated rates
    STATS_OUTPUT_PATH = 'bonds/datasets/tx_params.csv'  # output path for tx mean, std stats
    BATCH_LIST_PATH = 'bonds/datasets/batch_paths.csv'  # output path for tx file list
    BATCH_OUTPUT_DIR = 'bonds/batches'  # output dir for batches
    TS_META_COLS = [
        'ticker', 'entity_name', 'cik_code', 'sic_code',
        'naics_code', 'SICGroupMinorGroupTitle'
    ]
    CIK_CUSIP_COLS = ['CIK', 'CUSIP', 'CUSIP6']
    BOND_META_COLS = [
        'bond_sym_id', 'cusip_id', 'bond_sym_id', 'company_symbol', 'debt_type_cd',
        'issuer_nm', 'cpn_rt', 'cpn_type_cd',
        'trd_rpt_efctv_dt', 'mtrty_dt', 'cnvrb_fl', 'sub_prdct_type',
    ]
    INVALID_DATA_COLS = [
        'mtrty_dt', 'bond_sym_id', 'company_symbol', 'issuer_nm', 'scrty_ds',
        'close_pr', 'trans_dt', 'last_dt'
    ]
    ORDERED_MASTER_COLS = [
        'bond_sym_id', 'cusip_id', 'debt_type_cd', 'cpn_rt', 'trd_rpt_efctv_dt',
        'mtrty_dt'
    ]
    BOND_DATE_COLS = [
        'trd_rpt_efctv_dt', 'mtrty_dt'
    ]
    VALID_DEBT_TYPES = [
        '2LN-NT', 'B-BND', 'B-BNT', 'B-DEB', 'B-NT', 'BND', 'DEB', 'MTN',
        'OTH', 'OTH-BND', 'OTH-NT', 'OTH-OTH', 'S-BND', 'S-BNT',
        'S-DEB', 'S-NT', 'S-OTH', 'SB-NT', 'SBN-NT', 'SC-BND', 'SC-NT',
        'SC-OTH', 'SECNT', 'SR', 'SRDEB', 'SRNT', 'SRSEC', 'SRSUBNT',
        'SUBDEB', 'SUBNT', 'TGNT', 'UN-BND', 'UN-DEB', 'UN-NT', 'UNNT'
    ]
    SR_CODE_CVT = [
            'DEB', 'S-BND', 'S-DEB', 'SRDEB', 'SRNT', 'SR', 'UN-NT', 'UNNT',
            'UN-DEB'
        ]
    SUB_CODE_CVT = ['B-BND', 'B-NT', 'SUB-DEB', 'SUBDEB', 'SUBNT', 'SB-NT']
    VALID_NAICS = [
        '23', '31', '32', '33', '42', '44', '45', '48', '49', '51', '53',
        '56', '61', '71', '72', '78', '81'
    ]
    INVALID_NAICS = [
       '325412', '332994', '324199', '325110', '325414', '486210', '332117',
        '311942', '331319', '334516', '325510', '321213', '322211', '339113',
        '486990', '332116', '326121', '325211', '327332', '339114', '331319',
        '333314', '423450', '324110', '322291', '333132', '311942', '311611',
        '321920', '325320', '332313', '322121', '334517', '423310', '423510',
    ]
    INVALID_TICKERS = [
        'CLW', 'HOS'
    ]
    
    def __init__(self, is_local=False, base_url=None):
        # user provided base_url overrides is_local flag
        # is_local flag for local or s3 file location
        if base_url:
            # user defined mode
            self.base_url = base_url
            self._is_local = not self.base_url.startswith(self.S3_PREFIX)
        elif is_local:
            # local mode
            self._is_local = is_local
            self.base_url = self.LOCAL_BASE
        else:
            # remote s3 mode
            self._is_local = is_local
            self.base_url = join(self.S3_PREFIX, self.S3_BASE)
            
        # setup s3 connection
        self._conn = None
        if not self._is_local:
            self._conn = s3fs.S3FileSystem(anon=False)
            
    def fetch_tickers(self):
        """
            Generates list of cleaned bond tx tickers
            
            Returns:
            - list: ['ts1', 'ts2', ...]
        """
        ts_dir = join(self.base_url, self.BONDS_TXS_DIR)
        if self._is_local:
            ts = [fp for fp in listdir(ts_dir) if isfile(join(ts_dir, fp))]
        else:
            ts = self._conn.ls(ts_dir)
            
        ts = [fp.split('/')[-1] for fp in ts if fp.endswith('.csv')]
        ts = [t.split('.csv')[0] for t in ts]
        return ts
    
    def fetch_issuance_meta(self, ts, save=False):
        """
            Generates descriptive meta df for tickers.
            
            Returns:
            - dfout: pd.df
        """
        # fetch base meta data
        fpath = join(self.base_url, self.CIKS_META_PATH)
        if self._is_local:
            df = pd.read_csv(fpath, usecols=self.TS_META_COLS,
                            dtype=str)
        else:
            with self._conn.open(fpath, 'rb') as f:
                df = pd.read_csv(f, usecols=self.TS_META_COLS,
                                dtype=str)
        df = df[df.ticker.isin(ts)]
        
        # link ciks to cusips
        linkpath = join(self.base_url, self.CIK_CUSIP_LINK_PATH)
        if self._is_local:
            dflink = pd.read_csv(linkpath, usecols=self.CIK_CUSIP_COLS,
                                dtype=str)
        else:
            with self._conn.open(linkpath, 'rb') as f:
                dflink = pd.read_csv(f, usecols=self.CIK_CUSIP_COLS,
                                    dtype=str)
                
        # merge on cik code
        dfout = df.merge(dflink, how='inner', left_on='cik_code',
                        right_on='CIK')
        dfout = dfout.drop(columns=['CIK'])
        dfout = dfout.drop_duplicates(subset=['ticker', 'CUSIP', 'cik_code'])
        
        # fetch bond issuance meta data
        issuer_path = join(self.base_url, self.BOND_MASTER_PATH)
        ncols = [c for c in self.BOND_META_COLS if c not in self.BOND_DATE_COLS]
        col_dtypes = {k:str for k in ncols}
        col_dtypes['cpn_rt'] = np.float64
        if self._is_local:
            df = pd.read_csv(issuer_path,
                             usecols=self.BOND_META_COLS,
                             dtype=col_dtypes,
                             parse_dates=self.BOND_DATE_COLS)
        else:
            with self._conn.open(issuer_path, 'rb') as f:
                df = pd.read_csv(f,
                                 usecols=self.BOND_META_COLS,
                                 dtype=col_dtypes,
                                 parse_dates=self.BOND_DATE_COLS)

        # clean master file inputs
        df = df.dropna(subset=['cusip_id']).drop_duplicates(subset=['cusip_id'])
        
        # keep cpn_type_cd FXPV (plain vanilla fixed coupon) and OTH (other)
        df = df[(df.cpn_type_cd == 'FXPV') | (df.cpn_type_cd == 'OTH')]
        
        # exclude converts
        df = df[~(df.cnvrb_fl == 'Y')].drop(['cnvrb_fl'], axis=1)
        
        # only keep valid debt types
        # consolidate similar security and seniority
        df = df[df.debt_type_cd.isin(self.VALID_DEBT_TYPES)]
        df.loc[(df.debt_type_cd.isin(self.SR_CODE_CVT), 'debt_type_cd')] = 'S-NT'
        df.loc[(df.debt_type_cd.isin(self.SUB_CODE_CVT), 'debt_type_cd')] = 'SRSUBNT'
        
        # only coupon rates between 0.25% and 15%
        df = df[(df.cpn_rt >= 0.25) & (df.cpn_rt < 15)]
        
        # convert maturity from object to date and drop na
        df.mtrty_dt = pd.to_datetime(df.mtrty_dt, format='%Y%m%d', errors='coerce')
        df = df.dropna(subset=['mtrty_dt'])
        
        # only keep credits maturity after 2011-12-31
        df = df[df.mtrty_dt > '2011-12-31']
        
        # only corporate notes
        df = df[df.sub_prdct_type == 'CORP'].drop(columns=['sub_prdct_type'])
        
        # reorder columns
        df = df[self.ORDERED_MASTER_COLS]
        
        df['cusip6_id'] = df.cusip_id.astype(str).str[:6]
        
        # merge with input df on cusip
        dfout = dfout.merge(df, how='inner', left_on='CUSIP6',
                            right_on='cusip6_id')
        dfout = dfout.drop(columns=['cusip_id', 'cusip6_id'])
        
        # only keep valid naics and tickers
        dfout = dfout[(dfout.naics_code == 0) | (dfout.naics_code.str[:2].isin(self.VALID_NAICS))]
        dfout = dfout[~dfout.naics_code.isin(self.INVALID_NAICS)]
        dfout = dfout[~dfout.ticker.isin(self.INVALID_TICKERS)]
        dfout = dfout.drop_duplicates(subset=['ticker', 'CUSIP', 'cik_code'])
        
        # save ticker meta df
        if save:
            dtstr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
            outpath = 'tickers_'+dtstr+str(uuid4())+'.csv'
            save_path = join(self.base_url, self.TS_OUTPUT_DIR, outpath)
            dfout.to_csv(save_path, index=False)
    
        return dfout
    
    def build_dataset(self, tickers_path, nrows=5000, ntickers=None,
                      save=False):
        tickers_url = join(self.base_url, self.TS_OUTPUT_DIR, tickers_path)
        if self._is_local:
            dfts = pd.read_csv(tickers_url)
        else:
            with self._conn.open(tickers_url) as f:
                dfts = pd.read_csv(f)
                
        # get tickers and conditionally cap
        ts = dfts.ticker.unique()
        if ntickers is not None:
            ts = np.random.choice(ts, min(ntickers, ts.shape[0]))
            
        # pull raw tx data for tickers
        txs = []
        ps = [join(self.base_url, self.BONDS_TXS_DIR, f'{t}.csv') for t in ts]
        for t, p in zip(ts, ps):
            with self._conn.open(p) as f:
                df = pd.read_csv(f)
                # sample if record count above row cap
                if df.shape[0] > nrows:
                    df = df.sample(n=nrows)
                txs.append(df)

        df = pd.concat(txs, ignore_index=True)
        
        # scrub-a-dub
        # only keep valid debt types
        # consolidate similar security and seniority
        df = df[df.debt_type_cd.isin(self.VALID_DEBT_TYPES)]
        df.loc[(df.debt_type_cd.isin(self.SR_CODE_CVT), 'debt_type_cd')] = 'S-NT'
        df.loc[(df.debt_type_cd.isin(self.SUB_CODE_CVT), 'debt_type_cd')] = 'SRSUBNT'

        # save consolidated data
        if save:
            dtstr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
            outpath = 'txs_'+dtstr+str(uuid4())+".csv"
            save_path = join(self.base_url, self.DATA_OUTPUT_DIR, outpath)
            df.to_csv(save_path, index=False)

        return df
    
    def build_rates(self, save=False):
        # baml index data
        bml = join(self.base_url, self.RATES_BAML_DIR)
        ust = join(self.base_url, self.RATES_UST_DIR)
        if self._is_local:
            cps = [join(bml, f) for f in listdir(bml) if isfile(join(bml, f))]
            ups = [join(ust, f) for f in listdir(ust) if isfile(join(ust, f))]
        else:
            cps = [f for f in self._conn.ls(bml) if f.endswith('.csv')]
            ups = [f for f in self._conn.ls(ust) if f.endswith('.csv')]
        
        paths = cps + ups
        
        # merge each file
        p = paths[0]
        if self._is_local:
            df = pd.read_csv(p, index_col='DATE')
        else:
            with self._conn.open(p) as f:
                df = pd.read_csv(f, index_col='DATE')
        
        for p in paths[1:]:
            if self._is_local:
                tmp = pd.read_csv(p, index_col='DATE')
            else:
                with self._conn.open(p) as f:
                    tmp = pd.read_csv(f, index_col='DATE')
            df = df.join(tmp)
        
        df = df.replace(to_replace='.', value=np.nan).dropna()
        df = df.reset_index()

        # save consolidated data
        if save:
            dtstr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
            outpath = 'rates_'+dtstr+str(uuid4())+'.csv'
            save_path = join(self.base_url, self.RATES_OUTPUT_DIR, outpath)
            df.to_csv(save_path, index=False)
        return df
    
    def combine_txs_rates(self, txs_path, rates_path, save=False):
        tx_url = join(self.base_url, self.DATA_OUTPUT_DIR, txs_path)
        rates_url = join(self.base_url, self.RATES_OUTPUT_DIR, rates_path)
        if self._is_local:
            dftxs = pd.read_csv(tx_url, index_col='trans_dt')
            dfrates = pd.read_csv(rates_url, index_col='DATE')
        else:
            with self._conn.open(tx_url) as f:
                dftxs = pd.read_csv(f, index_col='trans_dt')
            with self._conn.open(rates_url) as f:
                dfrates = pd.read_csv(rates_url, index_col='trans_dt')
        dftxs = dftxs.join(dfrates).dropna().reset_index()
        dftxs = dftxs.rename(columns={'index': 'trans_dt'})
        dftxs = dftxs[dftxs.debt_type_cd == 'S-NT'].drop(['debt_type_cd'],
                                                         axis=1)
        
        # save consolidated data
        if save:
            dtstr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-')
            outpath = 'txs_rates_'+dtstr+str(uuid4())+'.csv'
            save_path = join(self.base_url, self.DATA_OUTPUT_DIR, outpath)
            dftxs.to_csv(save_path, index=False)
        return dftxs
    
    def load_data(self, data_path):
        data_url = join(self.base_url, self.DATA_OUTPUT_DIR, data_path)
        if self._is_local:
            cols = pd.read_csv(data_url, nrows=1).columns.values
            cols = [c for c in cols if c not in self.INVALID_DATA_COLS]
            df = pd.read_csv(data_url, usecols=cols, dtype=np.float64)
        else:
            with self._conn.open(data_url) as f:
                cols = pd.read_csv(f, nrows=1).columns.values
                cols = [c for c in cols if c not in self.INVALID_DATA_COLS]
                df = pd.read_csv(f, usecols=cols, dtype=np.float64)

        # drop columns with all zeros
        df = df.loc[:, (df != 0).any(axis=0)]
        
        # return with target last column
        cols = [c for c in df.columns.values if c != 'close_yld']
        cols.append('close_yld')
        return df[cols]

    def get_stats(self, df, save=False):
        dfstats = pd.DataFrame({
            'mean': df.mean(axis=0),
            'std': df.std(axis=0)
        })
        if save:
            save_path = join(self.base_url, self.STATS_OUTPUT_PATH)
            dfstats.to_csv(save_path, index=False)
        return dfstats
    
    def save_batch_paths(self):
        batch_dir = join(self.base_url, self.BATCH_OUTPUT_DIR)
        if self._is_local:
            ps = [p for p in listdir(batch_dir) if isfile(join(batch_dir, p))]
        else:
            ps = self._conn.ls(batch_dir)
            ps = [p.split('/')[-1] for p in ps if p.endswith('.csv')]

        df = pd.DataFrame(ps)
        df.to_csv(join(self.base_url, self.BATCH_LIST_PATH),
                  index=False, header=False)
        return df
    
    def batch_save(self, df, batch_cnt):
        batches = np.array_split(df, batch_cnt)
        for i, batch in enumerate(batches):
            save_path = join(self.base_url, self.BATCH_OUTPUT_DIR, f'batch_{i}.csv')
            batch.to_csv(save_path, index=False)