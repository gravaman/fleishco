from db.models.DB import Base
from db.models.Entity import Entity
from db.models.Corporate import Corporate
from db.models.CorpTx import CorpTx
from db.models.Financial import Financial
from db.models.EquityPx import EquityPx
from db.models.InterestRate import InterestRate


DATA_SOURCES = dict(
    entity='data/ciks/CIK_CUSIP.csv',
    entity_tickers='data/ciks/companies.csv',
    corporate='data/bonds/master_file.csv',
    financial='data/financials',
    interest_rate='data/rates/baml',
    equity_px='data/equities',
    corp_tx='data/bonds/clean_bond_close_pxs.csv'
)


def build(case='test'):
    if case == 'test':
        Base.metadata.drop_all()
        Base.metadata.create_all()
        insert_data(nrows=10000)
    elif case == 'prod':
        Base.metadata.drop_all()
        Base.metadata.create_all()
        insert_data()
    else:
        msg = "build case must be 'test' or 'prod'"
        raise ValueError(f'{msg}; received: {case}')


def insert_data(nrows=None):
    # restrict Corporate and CorpTx nrows to expedite data loading
    Entity.insert_entities(DATA_SOURCES['entity'],
                           DATA_SOURCES['entity_tickers'])
    Corporate.insert_corporates(DATA_SOURCES['corporate'],
                                nrows=nrows)
    CorpTx.insert_corp_txs(DATA_SOURCES['corp_tx'],
                           nrows=nrows)
    Financial.insert_financials(DATA_SOURCES['financial'])
    EquityPx.insert_equity_pxs(DATA_SOURCES['equity_px'])
    InterestRate.insert_interest_rates(DATA_SOURCES['interest_rate'])


build(case='prod')
