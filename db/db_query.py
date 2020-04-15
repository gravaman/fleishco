from datetime import timedelta
import pandas as pd
import numpy as np
from sqlalchemy import and_
from sqlalchemy.sql import func
from db.models.Corporate import Corporate  # noqa - needed for sqlalchemy table
from db.models.Entity import Entity  # noqa - needed for sqlalchemy table
from db.models.CorpTx import CorpTx  # noqa - needed for sqlalchemy table
from db.models.Financial import Financial, FS_ITEMS  # noqa - needed for sqlalchemy table
from db.models.EquityPx import EquityPx  # noqa - needed for sqlalchemy table
from db.models.InterestRate import InterestRate
from db.models.DB import db


NON_FIN_COLS = [
    'id', 'ticker', 'entity_id', 'earnings_release_date',
    'filing_date', 'period', 'period_start', 'period_end'
]

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
        'financingcashflow', 'paymentsofdividends', 'capex',
        'stockrepurchasedduringperiodvalue', 'adj_close',
        'stockrepurchasedduringperiodshares',
        'incometaxespaid', 'interestpaidnet',
        'sharesoutstandingendofperiod', 'restrictedcashandinvestmentscurrent',
        'paymentsofdividendspreferredstock', 'paymentsofdividendscommonstock',
        'paymentsofdividendsnoncontrollinginterest', 'assetimpairment',
        'restructuring', 'ev'
    ]

# residual columns
mkt_cap = (EquityPx.adj_close *
           Financial.sharesoutstandingendofperiod).label('mkt_cap')

ev = (mkt_cap +
      Financial.totaldebt -
      Financial.cash -
      Financial.shortterminvestments -
      Financial.longterminvestments).label('ev')

other_opex = (Financial.operatingexpenses -
              Financial.sgaexpense -
              Financial.researchanddevelopment -
              Financial.depreciationandamortizationexpense -
              Financial.operatingexpenseexitems
              ).label('other_opex')

other_investments = func.coalesce(Financial.totalinvestments -
                                  Financial.shortterminvestments -
                                  Financial.longterminvestments, 0
                                  ).label('other_investments')

other_current_assets = (Financial.currentassets -
                        Financial.shortterminvestments -
                        Financial.cash
                        ).label('other_current_assets')

other_lt_assets = (Financial.assets -
                   Financial.ppe -
                   Financial.longterminvestments -
                   Financial.currentassets
                   ).label('other_lt_assets')

other_opcf = (Financial.operatingcashflow -
              Financial.netincome -
              Financial.depreciationamortization -
              Financial.sharebasedcompensation -
              Financial.assetimpairment
              ).label('other_opcf')

other_invcf = (Financial.investingcashflow -
               Financial.capex -
               Financial.acquisitiondivestitures
               ).label('other_invcf')

dividends = (Financial.paymentsofdividendscommonstock +
             Financial.paymentsofdividendspreferredstock +
             Financial.paymentsofdividendsnoncontrollinginterest
             ).label('dividends')

other_fincf = (Financial.financingcashflow -
               Financial.paymentsofdividendscommonstock -
               Financial.paymentsofdividendspreferredstock -
               Financial.paymentsofdividendsnoncontrollinginterest
               ).label('other_fincf')

OTHER_COLS = ['other_opex', 'other_investments', 'other_current_assets',
              'other_lt_assets', 'other_opcf', 'other_invcf', 'dividends',
              'other_fincf', 'mkt_cap', 'ev']

FIN_COLS = ['adj_close'] + Financial.__table__.columns.keys() + OTHER_COLS


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
        'paymentsofdividendsnoncontrollinginterest', 'assetimpairment',
        'restructuring'
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


def get_corptx_ids(tickers, release_window, release_count, limit,
                   tick_limit, sd, ed):
    """
    Gets sample_count ids for ticker with at least release_count earnings
    within release_window.

    params
    ticker (str): corp_tx company_symbol
    release_window (int): relevant earnings_release_date window
    release_count (int): min number of earnings releases during release_window
    limit (int): samples return limit

    returns
    ids (1D np arr): matching ids
    """

    # subquery relevant corp_tx ids
    ctx_tick_stmt = db.query(CorpTx) \
        .filter(
            and_(
                CorpTx.company_symbol.in_(tickers),
                CorpTx.close_yld > 0,
                CorpTx.close_yld <= 20.0
            )) \
        .subquery('ctx_sq')

    # subquery corp_txs for release_count financial releases during window
    days_from_release = CorpTx.trans_dt-Financial.earnings_release_date
    fin_count = func.count(Financial.id).label('fin_count')

    window_stmt = db.query(CorpTx.id) \
        .distinct(CorpTx.cusip_id, CorpTx.trans_dt) \
        .join(ctx_tick_stmt, ctx_tick_stmt.c.id == CorpTx.id) \
        .join(Financial, Financial.ticker == CorpTx.company_symbol) \
        .filter(
            and_(
                days_from_release <= release_window,
                days_from_release > 0)) \
        .group_by(CorpTx.id) \
        .having(fin_count == release_count) \
        .subquery('window_sq')

    # partition by row number
    rn = func.row_number() \
        .over(partition_by=CorpTx.company_symbol,
              order_by=CorpTx.id).label('rn')

    sq = db.query(CorpTx.id, rn) \
        .join(window_stmt, CorpTx.id == window_stmt.c.id) \
        .join(EquityPx,
              and_(CorpTx.company_symbol == EquityPx.ticker,
                   CorpTx.trans_dt == EquityPx.date)) \
        .join(InterestRate, CorpTx.trans_dt == InterestRate.date) \
        .join(Financial, Financial.ticker == CorpTx.company_symbol) \
        .filter(
            and_(
                days_from_release <= release_window,
                days_from_release > 0,
                CorpTx.trans_dt <= ed,
                CorpTx.trans_dt > sd
            )).subquery('sq')

    s = db.query(CorpTx.id) \
        .distinct(CorpTx.id, CorpTx.trans_dt) \
        .join(sq, sq.c.id == CorpTx.id) \
        .filter(sq.c.rn <= tick_limit*release_count) \
        .order_by(CorpTx.trans_dt.asc()) \
        .limit(limit)

    ids = db.execute(s).fetchall()
    return np.unique(np.array(ids).flatten())


def get_target_stats(ids):
    """Returns mean, std (pop)"""
    s = db.query(func.avg(CorpTx.close_yld),
                 func.stddev_pop(CorpTx.close_yld)) \
        .filter(CorpTx.id.in_(ids))

    return db.execute(s).fetchall()[0]


def counts_by_sym(ids):
    s = db.query(CorpTx.company_symbol, func.count(CorpTx.company_symbol)) \
        .filter(CorpTx.id.in_(ids)) \
        .group_by(CorpTx.company_symbol)

    results = db.execute(s).fetchall()
    return results


def get_credit_data(ids, release_window, release_count, limit,
                    exclude_cols=[]):
    """
    Gets dataset consisting of interest rate, financial,
    credit terms, and yield data by transaction. Financial data
    is normalized by EV.

    params
    id (int): corp_tx id of credit
    release_window (int): relevant financials day window filter
    limit (int): samples limit

    returns
    fin (nd array): cleaned financials and equity price time series data
    tx (nd array): cleaned corp tx and interest rate data
    """
    financials = _get_financial_data(ids, release_window, release_count, limit)
    dffin = _clean_financial_data(financials, exclude_cols=exclude_cols)

    credit_txs = _get_credit_tx_data(ids)
    dftx = _clean_credit_tx_data(credit_txs)
    return dffin.values, dftx.values


def get_fin_cols(id, release_window, release_count, limit, exclude_cols=[]):
    financials = _get_financial_data([id],
                                     release_window,
                                     release_count,
                                     limit)
    dffin = _clean_financial_data(financials, exclude_cols=exclude_cols)
    return dffin.columns.values


def _get_financial_data(ids, release_window, release_count, limit):
    """
    Gets financials and equity px time series for corp_tx id

    params
    id (int): corp_tx id of credit
    release_window (int): relevant financials day window filter
    limit (int): samples limit

    returns
    samples (list): queried samples
    """
    # subquery relevant corp_tx ids
    ctx_tick_stmt = db.query(CorpTx.id) \
        .filter(
            and_(
                CorpTx.id.in_(ids),
                CorpTx.close_yld > 0
            )) \
        .subquery().alias('ctx_sq')

    # subquery corp_txs for release_count financial releases during window
    days_from_release = CorpTx.trans_dt-Financial.earnings_release_date
    fin_count = func.count(Financial.id).label('fin_count')

    window_stmt = db.query(CorpTx.id) \
        .join(ctx_tick_stmt, ctx_tick_stmt.c.id == CorpTx.id) \
        .join(Financial, Financial.ticker == CorpTx.company_symbol) \
        .filter(
            and_(
                days_from_release <= release_window,
                days_from_release > 0)) \
        .group_by(CorpTx.id) \
        .having(fin_count == release_count) \
        .subquery().alias('window_sq')

    # query financials with equity px and interest rate data
    s = db.query(EquityPx.adj_close, Financial, other_opex,
                 other_investments, other_current_assets, other_lt_assets,
                 other_opcf, other_invcf, dividends, other_fincf,
                 mkt_cap, ev) \
        .select_from(CorpTx) \
        .join(window_stmt, window_stmt.c.id == CorpTx.id) \
        .join(EquityPx,
              and_(CorpTx.company_symbol == EquityPx.ticker,
                   CorpTx.trans_dt == EquityPx.date)) \
        .join(InterestRate, CorpTx.trans_dt == InterestRate.date) \
        .join(Financial, Financial.ticker == CorpTx.company_symbol) \
        .filter(
            and_(
                days_from_release <= release_window,
                days_from_release > 0,
            )) \
        .distinct(CorpTx.cusip_id, CorpTx.trans_dt,
                  Financial.earnings_release_date) \
        .order_by(
            CorpTx.cusip_id,
            CorpTx.trans_dt.desc(),
            Financial.earnings_release_date.desc()) \
        .limit(limit)

    return db.execute(s).fetchall()


def _get_credit_tx_data(ids):
    """
    Gets credit samples for given corp_tx id

    params
    id (int): corp_tx id of credit

    returns
    samples (list): queried samples
    """
    days_to_mtrty = (CorpTx.mtrty_dt-CorpTx.trans_dt).label('days_to_mtrty')
    s = db.query(days_to_mtrty, CorpTx.close_yld,
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
        .filter(CorpTx.id.in_(ids)) \
        .join(InterestRate, CorpTx.trans_dt == InterestRate.date)

    return db.execute(s).fetchall()


def _clean_credit_tx_data(samples):
    """
    Cleans credit transaction and interest rate data

    params
    samples (list): raw credit tx and interest rate samples

    returns
    df (pd df): cleaned samples with last col target close_yld
    """
    # convert to df
    corp_tx_cols = ['days_to_mtrty', 'close_yld']
    rate_cols = [c for c in InterestRate.__table__.columns.keys()
                 if c not in ['id', 'date']]
    incols = corp_tx_cols + rate_cols
    df = pd.DataFrame(samples, columns=incols)

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

    # order cols interest rates, instrument metrics
    out_cols = rate_cols + ['days_to_mtrty', 'close_yld']
    return df[out_cols]


def _clean_financial_data(samples, exclude_cols=[]):
    """
    Cleans financial and equity px time series data

    params
    samples (list): raw financial and equity px time series data

    returns
    df (pd df): cleaned samples
    """
    # convert to df and drop non financial columns
    df = pd.DataFrame(samples,
                      columns=FIN_COLS).drop(labels=NON_FIN_COLS, axis=1)

    # set vals to float and fill na with 0
    df = pd.DataFrame(df.values,
                      columns=df.columns.values,
                      dtype=np.float64).fillna(0)

    # consolidate afs and sti
    df.shortterminvestments = np.where(
        df.availableforsalesecurities.eq(df.shortterminvestments),
        df.shortterminvestments,
        df.shortterminvestments + df.availableforsalesecurities)

    # normalize each row by ev and drop unnecessary cols
    df = df.div(df.ev, axis=0)
    df = df.drop(labels=DROP_COLS+exclude_cols, axis=1)
    return df


def _clean_credit_samples(samples):
    """
    Cleans credit samples

    params
    samples (list): raw credit samples

    returns
    df (pd df): cleaned samples with last col target close_yld
    """
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
    df['mkt_cap'] = df.adj_close * df.sharesoutstandingendofperiod
    df['ev'] = df.mkt_cap + df.totaldebt - df.cash \
        - df.shortterminvestments - df.longterminvestments
    xcol_mask = [c for c in df.columns.values
                 if c not in rate_cols + ['close_yld', 'days_to_mtrty']]
    df.loc[:, xcol_mask] = df.loc[:, xcol_mask].div(df.ev, axis=0)

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
        'paymentsofdividendsnoncontrollinginterest', 'assetimpairment',
        'restructuring'
    ]
    df = df.drop(labels=DROP_COLS, axis=1).dropna(axis=1, how='all')

    # order cols into financials, interest rates, instrument metrics
    outcols = [c for c in df.columns.values if c not in rate_cols +
               ['days_to_mtrty', 'close_yld']]
    outcols += rate_cols + ['days_to_mtrty', 'close_yld']
    df = df[outcols]
    return df
