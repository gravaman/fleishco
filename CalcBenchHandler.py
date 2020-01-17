import calcbench


calcbench.enable_backoff()


class CalcBenchHandler:
    DEI = ['earnings_release_date', 'period_end', 'period_start',
           'filing_date']
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

    def __init__(self, datadir='data'):
        self.datadir = datadir

    def all_metrics(self):
        return self.DEI+self.INS+self.BS+self.CFS

    def fetch_dates(self, company_identifiers=None, start_year=None,
                    start_period=None, end_year=None, end_period=None,
                    period_type=None):
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI
        return self._cb_fetch(**kwargs)

    def fetch_ins(self, company_identifiers=None, start_year=None,
                  start_period=None, end_year=None, end_period=None,
                  period_type=None):
        kwargs = locals()
        kwargs.pop('self', None)
        kwargs['metrics'] = self.DEI+self.INS
        return self._cb_fetch(**kwargs)

    def _cb_fetch(self, **kwargs):
        return calcbench.standardized_data(**kwargs)


if __name__ == '__main__':
    print(f'CalcBenchHandler Running')
    company_identifiers = ['FUN']

    if True:
        cb = CalcBenchHandler()
        df = cb.fetch_dates(company_identifiers=company_identifiers,
                            start_year=2018, start_period=1, end_year=2019,
                            end_period=4, period_type='quarterly')
        print(df)
