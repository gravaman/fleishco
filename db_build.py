from models.DB import Base
from models.Entity import Entity
from models.Corporate import Corporate
from models.CorpTx import CorpTx
from models.Financial import Financial
from models.EquityPx import EquityPx
from models.InterestRate import (
    InterestRate,
    BamlAAASYTW,
    BamlAASYTW,
    BamlASYTW,
    BamlBBBSYTW,
    BamlBBSYTW,
    BamlBSYTW,
    BamlCSYTW,
    Baml13YSTW,
    Baml35YSTW,
    Baml57YSTW,
    Baml710YSTW
)


DATA_SOURCES = dict(
    entity='data/ciks/CIK_CUSIP.csv',
    entity_tickers='data/ciks/companies.csv',
    corporate='data/bonds/master_file.csv',
    financial='data/financials',
    interest_rate=[
        (BamlAAASYTW, 'baml_aaa_sytw', 'data/rates/baml/BAMLC0A1CAAASYTW.csv'),
        (BamlAASYTW, 'baml_aa_sytw', 'data/rates/baml/BAMLC0A2CAASYTW.csv'),
        (BamlASYTW, 'baml_a_sytw', 'data/rates/baml/BAMLC0A3CASYTW.csv'),
        (BamlBBBSYTW, 'baml_bbb_sytw', 'data/rates/baml/BAMLC0A4CBBBSYTW.csv'),
        (BamlBBSYTW, 'baml_bb_sytw', 'data/rates/baml/BAMLH0A1HYBBSYTW.csv'),
        (BamlBSYTW, 'baml_b_sytw', 'data/rates/baml/BAMLH0A2HYBSYTW.csv'),
        (BamlCSYTW, 'baml_c_sytw', 'data/rates/baml/BAMLH0A3HYCSYTW.csv'),
        (Baml13YSTW, 'baml_13_sytw', 'data/rates/baml/BAMLC1A0C13YSYTW.csv'),
        (Baml35YSTW, 'baml_35_sytw', 'data/rates/baml/BAMLC2A0C35YSYTW.csv'),
        (Baml57YSTW, 'baml_57_sytw', 'data/rates/baml/BAMLC3A0C57YSYTW.csv'),
        (Baml710YSTW, 'baml_710_sytw', 'data/rates/baml/BAMLC4A0C710YSYTW.csv')
    ],
    equity_px='data/equities',
    corp_tx='data/bonds/clean_bond_close_pxs.csv'
)


def build(case='test'):
    if case == 'test':
        Base.metadata.drop_all()
        Base.metadata.create_all()
        insert_data(nrows=1000)
    elif case == 'prod':
        Base.metadata.drop_all()
        Base.metadata.create_all()
        insert_data()
    else:
        msg = "build case must be 'test' or 'prod'"
        raise ValueError(f'{msg}; received: {case}')


def insert_data(nrows=None):
    Entity.insert_entities(DATA_SOURCES['entity'],
                           DATA_SOURCES['entity_tickers'])
    Corporate.insert_corporates(DATA_SOURCES['corporate'],
                                nrows=nrows)
    CorpTx.insert_corp_txs(DATA_SOURCES['corp_tx'],
                           nrows=nrows)
    Financial.insert_financials(DATA_SOURCES['financial'],
                                nrows=nrows)
    EquityPx.insert_equity_pxs(DATA_SOURCES['equity_px'],
                               nrows=nrows)
    InterestRate.insert_interest_rates(DATA_SOURCES['interest_rate'],
                                       nrows=nrows)
    for cls, ticker, rates_path in DATA_SOURCES['interest_rate']:
        cls.insert_interest_rates(ticker, rates_path, nrows)


build(case='test')
