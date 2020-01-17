from pathlib import Path
import calcbench


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
            'mediumtermnotes', 'trustpreferredsecurities']

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
        self._dei_colnames = None

        # conditionally print settings to console
        if self.verbose:
            kwargs = locals()
            kwargs.pop('self', None)
            for k, v in kwargs.items():
                print(f'\t{k}: {v}')

    @property
    def dei_colnames(self):
        if self._dei_colnames is None:
            self._dei_colnames = self.DEI.copy()
            self._dei_colnames.insert(2, 'period')
        return self._dei_colnames.copy()

    def all_metrics(self):
        # all available standardized metrics
        return self.DEI+self.INS+self.BS+self.CFS

    def fetch_dates(self, company_identifiers=None, start_year=None,
                    start_period=None, end_year=None, end_period=None,
                    period_type=None):
        # build required args for calcbench api call
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI
        df = self._cb_fetch(**kwargs)

        # separate by company identifier
        cids = [cid.upper() for cid in company_identifiers]
        dfouts = [df.xs(cid, level=1, axis=1) for cid in cids]
        dfouts = [dfout.reset_index() for dfout in dfouts]
        dfouts = [dfout[self.dei_colnames] for dfout in dfouts]
        idx = 'earnings_release_date'
        dfouts = [dfout.set_index(idx) for dfout in dfouts]

        # conditionally save data
        if self.save_dir:
            self._check_dir()
            for (cid, dfout) in zip(cids, dfouts):
                path = f'{self.save_dir}/{cid}.csv'
                dfout.to_csv(path)
                if self.verbose:
                    print(f'dates saved to {path}')

        return dfouts

    def fetch_ins(self, company_identifiers=None, start_year=None,
                  start_period=None, end_year=None, end_period=None,
                  period_type=None):
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.INS
        return self._cb_fetch(**kwargs)

    def _cb_fetch(self, **kwargs):
        return calcbench.standardized_data(**kwargs)

    def _check_dir(self):
        if self._need_check_dir:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        self._need_check_dir = False


if __name__ == '__main__':
    print(f'CalcBenchHandler Running')
    company_identifiers = ['FUN', 'SIX']
    save_dir = 'data/financials'

    if True:
        cb = CalcBenchHandler(save_dir=save_dir)
        dfs = cb.fetch_dates(company_identifiers=company_identifiers,
                             start_year=2018, start_period=1, end_year=2019,
                             end_period=4, period_type='quarterly')
        for df in dfs:
            print(df)
