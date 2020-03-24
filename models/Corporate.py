import pandas as pd
from sqlalchemy import (
    Column, Integer, Float, String,
    Date, Boolean, ForeignKey
)
from sqlalchemy.orm import relationship
from models.DB import Base, db


class Corporate(Base):
    """
        Corporate issuance reference and key terms.

        finra_symbol:       FINRA issuance symbol (unique)
        cusip9:             CUSIP-9 issuance code (unique)
        entity:             Entity corresponding to issuer CUSIP-6
        bsym_id:            Bloomberg issuance symbol (unique)
        company_symbol:     Company symbol (typically ticker; non-unique)
        sub_prdct_type:     Issuance product type
        debt_type_cd:       FINRA debt type code
        issuer_nm:          Associated issuer name
        scrty_ds:           Seniority and security description
        cpn_rt:             Coupon rate
        cpn_type_cd:        Coupon payment code
        trd_rpt_efct_dt:    Effective trade start date
        mtrty_dt:           Maturity date
        ind_144a:           144a issuance indicator
    """
    __tablename__ = 'corporate'
    id = Column(Integer, primary_key=True)
    finra_symbol = Column(String(14), unique=True)
    cusip9 = Column(String(9), unique=True)
    entity_id = Column(Integer, ForeignKey('entity.id'))
    entity = relationship('Entity', back_populates='corporates')
    corp_txs = relationship('CorpTx', back_populates='corporate')
    bsym_id = Column(String(12), unique=True)
    company_symbol = Column(String(8))
    sub_prdct_type = Column(String(10))
    debt_type_cd = Column(String(8))
    issuer_nm = Column(String(80))
    scrty_ds = Column(String(80))
    cpn_rt = Column(Float)
    cpn_type_cd = Column(String(10))
    trd_rpt_efctv_dt = Column(Date)
    mtrty_dt = Column(Date)
    ind_144a = Column(Boolean)

    @classmethod
    def insert_corporates(cls, corps_path, nrows=None):
        # load data
        df = pd.read_csv(corps_path,
                         parse_dates=['trd_rpt_efctv_dt', 'mtrty_dt'],
                         nrows=nrows)
        cmap = {
            'bond_sym_id': 'finra_symbol',
            'cusip_id': 'cusip9'
        }
        df = df.rename(columns=cmap)

        # drop duplicate records and records missing required fields
        df = df.drop_duplicates(subset=['finra_symbol'])
        df = df.drop_duplicates(subset=['cusip9'])
        df = df.drop_duplicates(subset=['bsym_id'])
        df = df.dropna(subset=['finra_symbol', 'cusip9', 'bsym_id',
                               'cpn_rt', 'cpn_type_cd'])

        # only records with valid CUSIP-9 values
        df = df[df.cusip9.str.contains('^[A-Z0-9]{9}$')]

        # remove converts
        df = df[df.cnvrb_fl != 'Y']

        # remove unused fields, convert NaN to None, convert booleans
        df = df.drop(labels=['cnvrb_fl', 'dissem', 'grade'], axis=1)
        debt_type_cd = df.debt_type_cd.where(df.debt_type_cd.notnull(), None)
        df.loc[:, 'debt_type_cd'] = debt_type_cd
        df.loc[:, 'ind_144a'] = df.ind_144a == 'Y'

        # insert cleaned data into db table
        db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
        db.commit()
