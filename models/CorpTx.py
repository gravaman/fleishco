import pandas as pd
from sqlalchemy import (
    Column, Integer, Float, String,
    Date, ForeignKey
)
from sqlalchemy.orm import relationship
from models.DB import Base, db


class CorpTx(Base):
    __tablename__ = 'corp_tx'
    id = Column(Integer, primary_key=True)
    cusip_id = Column(String(9))
    bond_sym_id = Column(String(14))
    company_symbol = Column(String(20))
    corporate_id = Column(Integer, ForeignKey('corporate.id'))
    corporate = relationship('Corporate', back_populates='corp_txs')
    issuer_nm = Column(String(80))
    debt_type_cd = Column(String(8))
    scrty_ds = Column(String(80))
    cpn_rt = Column(Float)
    close_pr = Column(Float)
    close_yld = Column(Float)
    trans_dt = Column(Date)
    trd_rpt_efctv_dt = Column(Date)
    mtrty_dt = Column(Date)

    @classmethod
    def insert_corp_txs(cls, txs_path, nrows=None):
        df = pd.read_csv(txs_path, nrows=nrows).dropna()
        steps = df.shape[0] // 1000
        for step in range(steps):
            idx = step*1000
            if step == steps:
                dftxs = df.iloc[idx:]
                db.bulk_insert_mappings(cls, dftxs.to_dict(orient='records'))
                db.commit()
            else:
                dftxs = df.iloc[idx:idx+1000]
                db.bulk_insert_mappings(cls, dftxs.to_dict(orient='records'))
                db.commit()
