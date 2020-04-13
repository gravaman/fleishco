from datetime import datetime
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
from db.db_query import get_corptx_ids, get_credit_data


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


if __name__ == '__main__':
    print('-'*89)
    print('DB Benchmark Analysis')
    print('-'*89)
    # corp_tx_id_benchmark()
    tickers = ['QCOM', 'IBM']
    txids = get_corptx_ids(tickers, release_window=730,
                           release_count=8, limit=10000)
    txids = txids.tolist()
    # txids = [63370, 63369, 4916987, 4916986]
    # txids = [7098217]
    # txids = [63370]
    fin_data, ctx_data = get_credit_data(txids,
                                         release_window=730,
                                         release_count=8,
                                         limit=10000)
    print(len(fin_data), len(ctx_data))
