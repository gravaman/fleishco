from datetime import timedelta
import pandas as pd
import numpy as np
from sqlalchemy import and_
from sqlalchemy.sql import func
from models.Corporate import Corporate  # noqa - needed for sqlalchemy table
from models.Entity import Entity  # noqa - needed for sqlalchemy table
from models.CorpTx import CorpTx  # noqa - needed for sqlalchemy table
from models.Financial import Financial, FS_ITEMS  # noqa - needed for sqlalchemy table
from models.EquityPx import EquityPx  # noqa - needed for sqlalchemy table
from models.InterestRate import InterestRate
from models.DB import db


def build_dataset():
    # list of tickers and transaction dates
    s = db.query(CorpTx.company_symbol, CorpTx.trans_dt) \
            .filter(CorpTx.company_symbol == EquityPx.ticker) \
            .filter(CorpTx.trans_dt == EquityPx.date) \
            .group_by(CorpTx.company_symbol, CorpTx.trans_dt)
    ticks_and_dts = db.execute(s).fetchall()

    # find sample for n combos
    n = 1
    ticks_and_dts = ticks_and_dts[:n]
    for tick, dt in ticks_and_dts:
        fins = find_sample(tick, dt)
        if len(fins) > 0:
            print(fins)


def find_sample(ticker, trans_date, fin_count=8):
    # calculate minimum date with buffer for CY changes
    tdelta = timedelta(days=720)
    # min_dt = (trans_date-timedelta(days=days)).strftime('%Y-%m-%d')

    s = db.query(Financial.ticker, CorpTx.trans_dt,
                 Financial.earnings_release_date,
                 Financial.revenueadjusted) \
        .filter(and_(CorpTx.trans_dt == CorpTx.trans_dt,
                     CorpTx.close_pr <= 100)) \
        .filter(EquityPx.date == CorpTx.trans_dt) \
        .filter(CorpTx.company_symbol == EquityPx.ticker) \
        .filter(Financial.ticker == EquityPx.ticker) \
        .filter(Financial.earnings_release_date < CorpTx.trans_dt) \
        .filter(Financial.earnings_release_date >= CorpTx.trans_dt-tdelta) \
        .group_by(Financial.ticker, CorpTx.trans_dt,
                  Financial.earnings_release_date,
                  Financial.revenueadjusted) \
        .order_by(CorpTx.trans_dt.desc(),
                  Financial.earnings_release_date.asc()) \
        .limit(fin_count)

    return db.execute(s).fetchall()


def build_feature_data(day_window=100, sample_count=5, standardize=True):
    """
    Generates dataset consisting of interest rate, financial,
    credit terms, and yield data by transaction. Financial data
    is normalized by EV. All features are standardized across time.

    returns:
        - X (np arr): interest rate, financial, credit terms
        - Y (np arr): yield to worst
    """
    # query corporate transactions to generate terms, equity price, and
    # last reported financials within given day range
    s = db.query(CorpTx.company_symbol, CorpTx.trans_dt, CorpTx.mtrty_dt,
                 EquityPx.adj_close, CorpTx.close_yld, Financial,
                 InterestRate.BAMLC0A1CAAASYTW,
                 InterestRate.BAMLC0A2CAASYTW,
                 InterestRate.BAMLC0A3CASYTW,
                 InterestRate.BAMLC0A4CBBBSYTW,
                 InterestRate.BAMLH0A1HYBBSYTW,
                 InterestRate.BAMLH0A2HYBSYTW,
                 InterestRate.BAMLH0A3HYCSYTW,
                 InterestRate.BAMLC1A0C13YSYTW,
                 InterestRate.BAMLC2A0C35YSYTW,
                 InterestRate.BAMLC3A0C57YSYTW,
                 InterestRate.BAMLC4A0C710YSYTW,
                 InterestRate.BAMLC7A0C1015YSYTW,
                 InterestRate.BAMLC8A0C15PYSYTW) \
        .select_from(CorpTx) \
        .join(EquityPx,
              and_(CorpTx.company_symbol == EquityPx.ticker,
                   CorpTx.trans_dt == EquityPx.date)) \
        .join(InterestRate, CorpTx.trans_dt == InterestRate.date) \
        .join(Financial, CorpTx.company_symbol == Financial.ticker) \
        .filter(
            and_(
                CorpTx.trans_dt-Financial.earnings_release_date <= day_window,
                CorpTx.trans_dt-Financial.earnings_release_date > 0)) \
        .order_by(Financial.ticker,
                  CorpTx.trans_dt.desc(),
                  Financial.earnings_release_date.desc()) \
        .distinct(Financial.ticker, CorpTx.trans_dt) \
        .order_by(func.random()) \
        .limit(sample_count)

    samples = db.execute(s).fetchall()
    # convert to df
    colnames = ['company_symbol', 'trans_dt', 'mtrty_dt',
                'adj_close', 'close_yld']
    colnames += Financial.__table__.columns.keys()
    rate_cols = InterestRate.__table__.columns.keys()
    rate_cols = [c for c in rate_cols if c not in ['id', 'date']]
    colnames += [c for c in rate_cols if c not in ['id', 'date']]
    df = pd.DataFrame(samples, columns=colnames)

    # calculate days to maturity
    df['days_to_mtrty'] = (df.mtrty_dt-df.trans_dt)/np.timedelta64(1, 'D')

    # drop non-financial cols and fill nans with 0
    NON_FIN_COLS = [
        'company_symbol', 'id', 'ticker', 'entity_id', 'earnings_release_date',
        'filing_date', 'period', 'period_start', 'period_end', 'mtrty_dt',
        'trans_dt'
    ]
    df = df.drop(labels=NON_FIN_COLS, axis=1)
    df = pd.DataFrame(df.values,
                      columns=df.columns.values,
                      dtype=np.float64).fillna(0)

    # reduce complexity of rating based interest rate cols
    df.loc[:, 'BAMLH0A3HYCSYTW'] -= df.BAMLH0A2HYBSYTW
    df.loc[:, 'BAMLH0A2HYBSYTW'] -= df.BAMLH0A1HYBBSYTW
    df.loc[:, 'BAMLH0A1HYBBSYTW'] -= df.BAMLC0A4CBBBSYTW
    df.loc[:, 'BAMLC0A4CBBBSYTW'] -= df.BAMLC0A3CASYTW
    df.loc[:, 'BAMLC0A3CASYTW'] -= df.BAMLC0A2CAASYTW
    df.loc[:, 'BAMLC0A2CAASYTW'] -= df.BAMLC0A1CAAASYTW
    
    # reduce complexity of duration based interest rate cols
    df.loc[:, 'BAMLC8A0C15PYSYTW'] -= df.BAMLC7A0C1015YSYTW
    df.loc[:, 'BAMLC7A0C1015YSYTW'] -= df.BAMLC4A0C710YSYTW
    df.loc[:, 'BAMLC4A0C710YSYTW'] -= df.BAMLC3A0C57YSYTW
    df.loc[:, 'BAMLC3A0C57YSYTW'] -= df.BAMLC2A0C35YSYTW
    df.loc[:, 'BAMLC2A0C35YSYTW'] -= df.BAMLC1A0C13YSYTW
    
    # reduce complexity by adding line item complements
    # residual opex
    df['other_opex'] = df.operatingexpenses - df.sgaexpense \
        - df.researchanddevelopment - df.depreciationandamortizationexpense \
        - df.operatingexpenseexitems

    # residual addbacks
    df['other_addbacks'] = df.operatingexpenseexitems - df.restructuring \
        - df.assetimpairment

    # residual investments
    # consolidate afs and sti
    df.shortterminvestments = np.where(
        df.availableforsalesecurities.eq(df.shortterminvestments),
        df.shortterminvestments,
        df.shortterminvestments + df.availableforsalesecurities)

    df['other_investments'] = df.totalinvestments \
        - df.shortterminvestments - df.longterminvestments

    # residual current assets
    df['other_current_assets'] = df.currentassets - df.shortterminvestments \
        - df.cash

    # residual other long-term assets
    df['other_lt_assets'] = df.assets - df.ppe - df.longterminvestments \
        - df.currentassets

    # residual cash flow statement
    df['other_opcf'] = df.operatingcashflow - df.netincome \
        - df.depreciationamortization - df.sharebasedcompensation \
        - df.assetimpairment

    df['other_invcf'] = df.investingcashflow - df.capex \
        - df.acquisitiondivestitures

    df['dividends'] = df.paymentsofdividendscommonstock \
        + df.paymentsofdividendspreferredstock \
        + df.paymentsofdividendsnoncontrollinginterest \

    df['other_fincf'] = df.financingcashflow \
        - df.dividends \
        - df.paymentsforrepurchaseofcommonstock

    # capitalization adjustments:
    # [1] calculate mkt cap and ev
    # [2] normalize each row by ev
    # [3] conditionally standardize each column
    df['mkt_cap'] = df.adj_close * df.sharesoutstandingendofperiod
    df['ev'] = df.mkt_cap + df.totaldebt - df.cash \
        - df.shortterminvestments - df.longterminvestments
    xcol_mask = [c for c in df.columns.values
                 if c not in rate_cols + ['close_yld', 'days_to_mtrty']]
    df.loc[:, xcol_mask] = df.loc[:, xcol_mask].div(df.ev, axis=0)
    if standardize:
        df = (df - df.mean(axis=0)) / df.std(axis=0)

    # drop unnecessary cols
    DROP_COLS = [
        'avgsharesoutstandingbasic', 'avgdilutedsharesoutstanding',
        'commonstockdividendspershare', 'operatingexpenses',
        'operatingexpenseexitems', 'operatingincome', 'ebitda',
        'earningsbeforetaxes', 'netincome', 'totalinvestments',
        'availableforsalesecurities', 'currentassets', 'assets',
        'currentlongtermdebt', 'longtermdebt',
        'lineofcreditfacilityamountoutstanding', 'secureddebt',
        'convertibledebt', 'termloan', 'mortgagedebt', 'unsecureddebt',
        'mediumtermnotes', 'trustpreferredsecurities', 'seniornotes',
        'subordinateddebt', 'operatingcashflow', 'investingcashflow',
        'financingcashflow', 'paymentsofdividends', 'capex', 'ev',
        'stockrepurchasedduringperiodvalue', 'adj_close',
        'stockrepurchasedduringperiodshares',
        'incometaxespaid', 'interestpaidnet',
        'sharesoutstandingendofperiod', 'restrictedcashandinvestmentscurrent',
        'paymentsofdividendspreferredstock', 'paymentsofdividendscommonstock',
        'paymentsofdividendsnoncontrollinginterest'
    ]
    df = df.drop(labels=DROP_COLS, axis=1).dropna(axis=1, how='all')

    # order cols into financials, interest rates, instrument metrics
    outcols = [c for c in df.columns.values if c not in rate_cols +
               ['days_to_mtrty', 'close_yld']]
    outcols += rate_cols + ['days_to_mtrty']

    # split into x, y, column names
    # outcols = [c for c in df.columns.values if c != 'close_yld']
    x, y = df[outcols].values, df.close_yld.values
    return x, y, outcols