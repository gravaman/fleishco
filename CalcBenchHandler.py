from pathlib import Path
import calcbench
import pandas as pd


calcbench.enable_backoff()


class CalcBenchHandler:
    DEI = ['earnings_release_date', 'filing_date', 'period_start',
           'period_end']
    INS = ['revenueadjusted', 'grossprofit', 'sgaexpense',
           'researchanddevelopment', 'operatingexpenses',
           'operatingexpenseexitems', 'assetimpairment',
           'restructuring', 'depreciationandamortizationexpense',
           'operatingincome', 'ebitda', 'interestexpense',
           'earningsbeforetaxes', 'incometaxes', 'netincome',
           'avgsharesoutstandingbasic', 'avgdilutedsharesoutstanding',
           'commonstockdividendspershare']
    BS = ['cash', 'restrictedcashandinvestmentscurrent',
          'availableforsalesecurities', 'shortterminvestments',
          'longterminvestments', 'totalinvestments',
          'currentassets', 'currentliabilities', 'ppe', 'assets',
          'currentlongtermdebt', 'longtermdebt', 'totaldebt',
          'sharesoutstandingendofperiod',
          'lineofcreditfacilityamountoutstanding', 'secureddebt',
          'seniornotes', 'subordinateddebt', 'convertibledebt',
          'termloan', 'mortgagedebt', 'unsecureddebt',
          'mediumtermnotes', 'trustpreferredsecurities']
    CFS = ['depreciationamortization', 'operatingcashflow',
           'sharebasedcompensation', 'capexgross',
           'capitalassetsales', 'capex', 'acquisitiondivestitures',
           'investingcashflow', 'paymentsofdividends',
           'paymentsofdividendscommonstock',
           'paymentsofdividendspreferredstock',
           'paymentsofdividendsnoncontrollinginterest',
           'financingcashflow', 'stockrepurchasedduringperiodshares',
           'stockrepurchasedduringperiodvalue',
           'paymentsforrepurchaseofcommonstock', 'incometaxespaid',
           'interestpaidnet']
    DEBT = ['currentlongtermdebt', 'longtermdebt', 'totaldebt',
            'lineofcreditfacilityamountoutstanding', 'secureddebt',
            'seniornotes', 'subordinateddebt', 'convertibledebt',
            'termloan', 'mortgagedebt', 'unsecureddebt',
            'mediumtermnotes', 'trustpreferredsecurities',
            'commercialpaper']
    STD_MET = ['DEI', 'INS', 'BS', 'CFS']

    def __init__(self, save_dir=None, verbose=True):
        # sanity check
        if save_dir and not isinstance(save_dir, str):
            msg = (
                f'CalcBenchHandler save_dir must be str, '
                f'given {save_dir} ({type(save_dir)})'
            )
            raise ValueError(msg)

        # save settings
        self.save_dir = save_dir
        self.verbose = verbose
        self._need_check_dir = True if save_dir else False
        self._all_colnames = None
        self._dei_colnames = None
        self._ins_colnames = None
        self._bs_colnames = None
        self._cfs_colnames = None
        self._debt_colnames = None

        # conditionally print settings to console
        if self.verbose:
            kwargs = locals()
            kwargs.pop('self', None)
            for k, v in kwargs.items():
                print(f'\t{k}: {v}')

    @property
    def all_colnames(self):
        if self._all_colnames is None:
            self._all_colnames = self.all_metrics()
            self._all_colnames.insert(2, 'period')
        return self._all_colnames.copy()

    @property
    def dei_colnames(self):
        if self._dei_colnames is None:
            self._dei_colnames = self.DEI.copy()
            self._dei_colnames.insert(2, 'period')
        return self._dei_colnames.copy()

    @property
    def ins_colnames(self):
        if self._ins_colnames is None:
            self._ins_colnames = self.dei_colnames+self.INS.copy()
        return self._ins_colnames.copy()

    @property
    def bs_colnames(self):
        if self._bs_colnames is None:
            self._bs_colnames = self.dei_colnames+self.BS.copy()
        return self._bs_colnames.copy()

    @property
    def cfs_colnames(self):
        if self._cfs_colnames is None:
            self._cfs_colnames = self.dei_colnames+self.CFS.copy()
        return self._cfs_colnames.copy()

    @property
    def debt_colnames(self):
        if self._debt_colnames is None:
            self._debt_colnames = self.dei_colnames+self.DEBT.copy()
        return self._debt_colnames.copy()

    def all_metrics(self):
        # all available standardized metrics
        outs = []
        for k in self.STD_MET:
            mets = getattr(self, k)
            outs += mets.copy()
        return outs

    def fetch_dates(self, company_identifiers=None, start_year=None,
                    start_period=None, end_year=None, end_period=None,
                    period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.dei_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_ins(self, company_identifiers=None, start_year=None,
                  start_period=None, end_year=None, end_period=None,
                  period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.INS

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.ins_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_bs(self, company_identifiers=None, start_year=None,
                 start_period=None, end_year=None, end_period=None,
                 period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.BS

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.bs_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_cfs(self, company_identifiers=None, start_year=None,
                  start_period=None, end_year=None, end_period=None,
                  period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.CFS

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.cfs_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_debt(self, company_identifiers=None, start_year=None,
                   start_period=None, end_year=None, end_period=None,
                   period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.DEBT

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.debt_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_all(self, company_identifiers=None, start_year=None,
                  start_period=None, end_year=None, end_period=None,
                  period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.all_metrics()

        # make req, clean res, conditionally save
        df = self._cb_fetch(**kwargs)
        dfouts = self._split_by_cid(df, colnames=self.all_colnames)

        if self.save_dir:
            self._save_data(dfouts)

        return dfouts

    def fetch_ciks(self, ticker_path):
        # fetch ciks from tickers
        tickers = pd.read_csv(ticker_path).ticker.values
        if self.verbose:
            print(f'fetching ciks for {tickers}')
        kwargs = dict(company_identifiers=tickers,
                      metrics=['ticker', 'centralindexkey'])

        # clean res to recombine ciks
        df = self._cb_fetch(**kwargs)
        df = df.reset_index()
        prefix = df.period.astype(str).values
        cikser = df.stack().droplevel(0).centralindexkey.dropna()
        ciks = prefix+cikser.values.astype(str)
        ciks = [cik.split('.', 1)[0] for cik in ciks]
        data = {'ticker': df.ticker.values[0], 'cik': ciks}
        dfciks = pd.DataFrame(data).set_index('ticker')

        if self.save_dir:
            self._check_dir()
            path = f'{self.save_dir}/equities.csv'
            dfciks.to_csv(path)
            if self.verbose:
                print(f'data saved to {path}')

        return dfciks

    def _cb_fetch(self, **kwargs):
        return calcbench.standardized_data(**kwargs)

    def _split_by_cid(self, df, colnames=None):
        # separate by company identifier and name each new df by cid
        cids = list(set([cid for _, cid in df.columns]))
        dfouts = [df.xs(cid, level=1, axis=1) for cid in cids]
        dfouts = [dfout.reset_index() for dfout in dfouts]
        if colnames:
            dfouts = [dfout[colnames] for dfout in dfouts]

        # set index; must happen before name
        idx = 'earnings_release_date'
        dfouts = [dfout.set_index(idx) for dfout in dfouts]

        # add names for identification
        for (cid, dfout) in zip(cids, dfouts):
            print('adding cid as name:', cid)
            dfout.name = cid
        return dfouts

    def _save_data(self, dfouts):
        self._check_dir()
        for dfout in dfouts:
            path = f'{self.save_dir}/{dfout.name}.csv'
            dfout.to_csv(path)
            if self.verbose:
                print(f'data saved to {path}')

    def _cids_from_df(self, df):
        return list(set([cid for _, cid in df.columns]))

    def _check_dir(self):
        if self._need_check_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self._need_check_dir = False


if __name__ == '__main__':
    print(f'CalcBenchHandler Running')
    company_identifiers = ['0000811532', '0000701374']
    save_dir = 'data/financials'
    cik_dir = 'data/ciks'
    ticker_path = 'data/ciks/tickers.csv'

    if False:
        cbh = CalcBenchHandler(save_dir=save_dir)
        dfs = cbh.fetch_ins(company_identifiers=company_identifiers,
                            start_year=2018, start_period=1, end_year=2019,
                            end_period=4, period_type='quarterly')
        for df in dfs:
            print(f'{df.name}\n{df}')

    if False:
        cbh = CalcBenchHandler(save_dir=cik_dir)
        df = cbh.fetch_ciks(ticker_path)
        print(df)
