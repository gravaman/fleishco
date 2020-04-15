from datetime import datetime
from tabulate import tabulate
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
from ml.models.CreditDataset import CreditDataset
from db.db_query import get_corptx_ids, _get_financial_data


TECH = [
    'AAPL', 'MSFT', 'INTC', 'IBM', 'QCOM', 'ORCL', 'TXN', 'MU', 'AMZN', 'GOOG',
    'NVDA', 'JNPR', 'ADI', 'ADBE', 'STX', 'AVT', 'ARW', 'KLAC', 'A', 'NTAP',
    'VRSK', 'TECD', 'MRVL', 'KEYS'
]


def default_corptx_ids(tickers, release_window=730, release_count=8,
                       limit=100):
    """Queries corp_tx table for each individual id.

    Baseline method used for benchmarking corp_tx id queries.

    Args:
        tickers: A list of tickers mapping to corp_tx company_symbol field.
        release_window: An optional variable for the relevant financial
            earnings release date window.
        release_count: An optional variable for the requisite min number of
            filings within the release_window.
        limit: An option variable for the max number of ids to return.

    Returns:
        A list of corp_tx ids corresponding to matching corp_tx records. For
        example:
            [1034, 7896, 6732]
    """
    days_from_release = CorpTx.trans_dt-Financial.earnings_release_date
    ids = []
    for ticker in tickers:
        s = db.query(CorpTx.id) \
            .select_from(CorpTx) \
            .filter(CorpTx.company_symbol == ticker) \
            .join(
                EquityPx,
                and_(CorpTx.company_symbol == EquityPx.ticker,
                     CorpTx.trans_dt == EquityPx.date)) \
            .join(InterestRate, CorpTx.trans_dt == InterestRate.date) \
            .join(Financial, CorpTx.company_symbol == Financial.ticker) \
            .filter(
                and_(
                    days_from_release <= release_window,
                    days_from_release > 0,
                    CorpTx.close_yld > 0)) \
            .distinct(CorpTx.cusip_id, CorpTx.company_symbol,
                      CorpTx.trans_dt) \
            .having(
                func.count(Financial.earnings_release_date) >= release_count
            ).group_by(CorpTx.id) \
            .order_by(CorpTx.trans_dt.asc())

        if limit is not None:
            s = s.limit(limit)

        ids.append(db.execute(s).fetchall())

    ids = np.array(ids).flatten()
    return np.unique(ids)


def batch_corptx_ids(tickers, release_window=730, release_count=8,
                     limit=100):
    """Batch queries corp_tx table for given ids.

    Batch implementation of corp_tx id query.

    Args:
        tickers: A list of tickers mapping to corp_tx company_symbol field.
        release_window: An optional variable for the relevant financial
            earnings release date window.
        release_count: An optional variable for the requisite min number of
            filings within the release_window.
        limit: An option variable for the max number of ids to return.

    Returns:
        A list of corp_tx ids corresponding to matching corp_tx records. For
        example:
            [1034, 7896, 6732]
    """
    days_from_release = CorpTx.trans_dt-Financial.earnings_release_date
    s = db.query(CorpTx.id) \
        .select_from(CorpTx) \
        .filter(CorpTx.company_symbol.in_(tickers)) \
        .join(
            EquityPx,
            and_(CorpTx.company_symbol == EquityPx.ticker,
                 CorpTx.trans_dt == EquityPx.date)) \
        .join(InterestRate, CorpTx.trans_dt == InterestRate.date) \
        .join(Financial, CorpTx.company_symbol == Financial.ticker) \
        .filter(
            and_(
                days_from_release <= release_window,
                days_from_release > 0,
                CorpTx.close_yld > 0)) \
        .distinct(CorpTx.cusip_id, CorpTx.company_symbol,
                  CorpTx.trans_dt) \
        .having(
            func.count(Financial.earnings_release_date) >= release_count
        ).group_by(CorpTx.id) \
        .order_by(CorpTx.trans_dt.asc()) \
        .limit(limit)

    ids = db.execute(s).fetchall()
    return np.array(ids).flatten()


def corp_tx_id_benchmark():
    tickers = ['AAPL', 'MSFT', 'IBM', 'QCOM', 'INTC']
    print('-'*89)
    print(f'[1] corp_tx id benchmark test')
    start = datetime.now()
    baseline_ids = default_corptx_ids(tickers, limit=100)
    baseline_ms = (datetime.now()-start).microseconds/1000
    print(f'-- baseline: {baseline_ms: 5.1f} ms '
          f'| {len(baseline_ids): 5.0f} ids')

    start = datetime.now()
    batch_ids = batch_corptx_ids(tickers, limit=100*len(tickers))
    batch_ms = (datetime.now()-start).microseconds/1000
    print(f'-- batch: {batch_ms: 5.1f} ms '
          f'| {len(batch_ids): 5.0f} ids')


def aggregate_summary(tickers, release_window=730, T=8):
    print(f'ticker list length: {len(tickers)}')

    # get id list
    idxs = get_corptx_ids(tickers,
                          release_window=release_window,
                          release_count=T, limit=None
                          )
    print(f'id list length: {len(idxs)}')

    # get financials
    fins = _get_financial_data(idxs.tolist(),
                               release_window=release_window,
                               release_count=T,
                               limit=None)
    print(f'fins list length: {len(fins)} vs expected of {len(idxs)*T}')

    total_txs = len(idxs)
    total_dataset = CreditDataset(tickers,
                                  T=T,
                                  standardize=False)
    print(f'dataset length: {len(total_dataset)}')

    sample = next(iter(total_dataset))
    xfin_sz, xctx_sz, y_sz = [t.size() for t in sample]

    print('-'*89)
    print(f'Aggregate Sample Summary')
    print('-'*89)

    table = [
        ['tickers', f'{len(tickers):,}'],
        ['X_fin size', xfin_sz],
        ['X_ctx size', xctx_sz],
        ['Y size', y_sz]
    ]
    print(tabulate(table))
    print('-'*89)

    return total_txs


def ticker_summary(tickers, total_txs, T=8):
    rows = []
    headers = ['ticker', 'tx count', '% txs']
    for ticker in tickers:
        dataset = CreditDataset([ticker],
                                T=T,
                                standardize=False,
                                )
        tx_count = len(dataset)
        tx_pct = tx_count/total_txs*100
        row = [ticker, f'{tx_count:6,}', tx_pct]
        rows.append(row)

    total_row = ['TOTAL', f'{total_txs:,}', None]
    rows.append(total_row)
    print(tabulate(rows, headers=headers, floatfmt='.1f'))


def summarize_data(tickers):
    release_window = 730
    T = 8
    total_txs = aggregate_summary(tickers,
                                  release_window=release_window,
                                  T=T)
    ticker_summary(tickers, total_txs, T)


if __name__ == '__main__':
    print('-'*89)
    print('Data Summary Analysis')
    print('-'*89)
    idxs = get_corptx_ids(TECH[:2], release_window=730, release_count=8,
                          limit=11000, tick_limit=5000)
    print(f'idxs length: {len(idxs)}')
