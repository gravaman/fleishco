from os.path import join
import pandas as pd
import numpy as np
from sqlalchemy import (
    Column, Integer, Float, String,
    Date, ForeignKey
)
from db.models.DB import Base, db
from db.models.utils import get_tickers


FS_ITEMS = [
    'revenueadjusted', 'grossprofit', 'sgaexpense', 'researchanddevelopment',
    'operatingexpenses', 'operatingexpenseexitems', 'assetimpairment',
    'restructuring', 'depreciationandamortizationexpense', 'operatingincome',
    'ebitda', 'interestexpense', 'earningsbeforetaxes', 'incometaxes',
    'netincome', 'avgsharesoutstandingbasic', 'avgdilutedsharesoutstanding',
    'commonstockdividendspershare', 'cash',
    'restrictedcashandinvestmentscurrent', 'availableforsalesecurities',
    'shortterminvestments', 'longterminvestments', 'totalinvestments',
    'currentassets', 'currentliabilities', 'ppe', 'assets',
    'currentlongtermdebt', 'longtermdebt', 'totaldebt',
    'sharesoutstandingendofperiod', 'lineofcreditfacilityamountoutstanding',
    'secureddebt', 'seniornotes', 'subordinateddebt', 'convertibledebt',
    'termloan', 'mortgagedebt', 'unsecureddebt', 'mediumtermnotes',
    'trustpreferredsecurities', 'depreciationamortization',
    'operatingcashflow', 'sharebasedcompensation', 'capexgross',
    'capitalassetsales', 'capex', 'acquisitiondivestitures',
    'investingcashflow', 'paymentsofdividends',
    'paymentsofdividendscommonstock', 'paymentsofdividendspreferredstock',
    'paymentsofdividendsnoncontrollinginterest', 'financingcashflow',
    'stockrepurchasedduringperiodshares', 'stockrepurchasedduringperiodvalue',
    'paymentsforrepurchaseofcommonstock', 'incometaxespaid', 'interestpaidnet'
]


class Financial(Base):
    __tablename__ = 'financial'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    entity_id = Column(Integer,
                       ForeignKey('entity.id'),
                       index=True)
    earnings_release_date = Column(Date, index=True)
    filing_date = Column(Date)
    period = Column(String(6))
    period_start = Column(Date)
    period_end = Column(Date)
    revenueadjusted = Column(Float)
    grossprofit = Column(Float)
    sgaexpense = Column(Float)
    researchanddevelopment = Column(Float)
    operatingexpenses = Column(Float)
    operatingexpenseexitems = Column(Float)
    assetimpairment = Column(Float)
    restructuring = Column(Float)
    depreciationandamortizationexpense = Column(Float)
    operatingincome = Column(Float)
    ebitda = Column(Float)
    interestexpense = Column(Float)
    earningsbeforetaxes = Column(Float)
    incometaxes = Column(Float)
    netincome = Column(Float)
    avgsharesoutstandingbasic = Column(Float)
    avgdilutedsharesoutstanding = Column(Float)
    commonstockdividendspershare = Column(Float)
    cash = Column(Float)
    restrictedcashandinvestmentscurrent = Column(Float)
    availableforsalesecurities = Column(Float)
    shortterminvestments = Column(Float)
    longterminvestments = Column(Float)
    totalinvestments = Column(Float)
    currentassets = Column(Float)
    currentliabilities = Column(Float)
    ppe = Column(Float)
    assets = Column(Float)
    currentlongtermdebt = Column(Float)
    longtermdebt = Column(Float)
    totaldebt = Column(Float)
    sharesoutstandingendofperiod = Column(Float)
    lineofcreditfacilityamountoutstanding = Column(Float)
    secureddebt = Column(Float)
    seniornotes = Column(Float)
    subordinateddebt = Column(Float)
    convertibledebt = Column(Float)
    termloan = Column(Float)
    mortgagedebt = Column(Float)
    unsecureddebt = Column(Float)
    mediumtermnotes = Column(Float)
    trustpreferredsecurities = Column(Float)
    depreciationamortization = Column(Float)
    operatingcashflow = Column(Float)
    sharebasedcompensation = Column(Float)
    capexgross = Column(Float)
    capitalassetsales = Column(Float)
    capex = Column(Float)
    acquisitiondivestitures = Column(Float)
    investingcashflow = Column(Float)
    paymentsofdividends = Column(Float)
    paymentsofdividendscommonstock = Column(Float)
    paymentsofdividendspreferredstock = Column(Float)
    paymentsofdividendsnoncontrollinginterest = Column(Float)
    financingcashflow = Column(Float)
    stockrepurchasedduringperiodshares = Column(Float)
    stockrepurchasedduringperiodvalue = Column(Float)
    paymentsforrepurchaseofcommonstock = Column(Float)
    incometaxespaid = Column(Float)
    interestpaidnet = Column(Float)

    @classmethod
    def insert_financials(cls, fin_dir, nrows=None):
        tickers = get_tickers(fin_dir)
        fpaths = [join(fin_dir, f'{t}.csv') for t in tickers]
        for ticker, fin_path in zip(tickers, fpaths):
            df = pd.read_csv(fin_path, nrows=nrows).dropna(
                subset=['earnings_release_date',
                        'filing_date'])
            if df.shape[0] > 0:
                df = df.replace(to_replace={np.nan: None})
                df['ticker'] = ticker
                db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
                db.commit()
