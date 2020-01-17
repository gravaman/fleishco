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

    def __init__(self, datadir='data'):
        self.datadir = datadir

    def all_metrics(self):
        return self.DEI+self.INS


if __name__ == '__main__':
    symbols = ['FUN']
    mets = ['revenue']
    df = calcbench.standardized_data(company_identifiers=symbols, metrics=mets,
                                     start_year=2018, start_period=1,
                                     end_year=2019, end_period=4,
                                     period_type='quarterly')
    print(df)
