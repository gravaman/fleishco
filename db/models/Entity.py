import pandas as pd
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from db.models.DB import Base, db


class Entity(Base):
    """
        The entity model provides corporate reference information.

        cusip6:     The CUSIP-6 base code which uniquely identifies issuer.
        cik:        SEC Central Index Key which uniquely identifies a filer.
                    This does not uniquely identify an issuer as a filer may
                    have multiple issuers, each with multiple issuances. The
                    cusip6 code is used to uniquly identify the issuer.
        naics:      NAICS business establishment code
        sic:        SIC industry code
        ticker:     related equity ticker
        name:       corporate entity name associated with the issuer
        sic_mtitle: SIC minor group title
        corporates:  corporate issuance relationshiup
    """
    __tablename__ = 'entity'
    id = Column(Integer, primary_key=True)
    cusip6 = Column(String(6), unique=True)
    cik = Column(String(10))
    naics = Column(String(6))
    sic = Column(String(4))
    ticker = Column(String(10))
    name = Column(String(120))
    sic_mtitle = Column(String(80))
    corporates = relationship('Corporate',
                              cascade='all, delete-orphan')
    financials = relationship('Financial',
                              cascade='all, delete-orphan')
    equity_pxs = relationship('EquityPx',
                              cascade='all, delete-orphan')

    @classmethod
    def insert_entities(cls, entity_path, tickers_path):
        # load raw entity data
        target_cols = ['CIK', 'SEC_Name', 'CUSIP6']
        dtypes = {'CIK': str, 'SEC_Name': str, 'CUSIP6': str}
        df = pd.read_csv(entity_path,
                         usecols=target_cols,
                         dtype=dtypes)
        ticks_target_cols = ['ticker', 'sic_code', 'naics',
                             'cik_code', 'SICGroupMinorGroupTitle']
        ticks_dtypes = {k: str for k in ticks_target_cols}
        dfticks = pd.read_csv(tickers_path,
                              usecols=ticks_target_cols,
                              dtype=ticks_dtypes)
        df = df.merge(dfticks, left_on='CIK', right_on='cik_code')
        df = df.drop(labels=['cik_code'], axis=1)
        df = df[df.ticker.str.len() <= 10]

        # update column names, filter valid cusips and drop duplicates
        colnames = {
            'CIK': 'cik',
            'SEC_Name': 'name',
            'CUSIP6': 'cusip6',
            'sic_code': 'sic',
            'SICGroupMinorGroupTitle': 'sic_mtitle'
        }
        df = df.rename(columns=colnames)
        df = df[df.cusip6.str.contains('^[A-Z0-9]{6}$')]
        df = df.drop_duplicates(subset=['cusip6'])

        # insert data into db
        db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
        db.commit()
