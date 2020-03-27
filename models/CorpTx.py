import pandas as pd
from sqlalchemy import (
    Column, Integer, Float, String,
    Date, ForeignKey
)
from models.DB import Base, db


class CorpTx(Base):
    __tablename__ = 'corp_tx'
    id = Column(Integer, primary_key=True)
    cusip_id = Column(String(9))
    bond_sym_id = Column(String(14))
    company_symbol = Column(String(20))
    corporate_id = Column(Integer,
                          ForeignKey('corporate.id'),
                          index=True)
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
        if nrows is None:
            step_size = 100000
        else:
            step_size = min(nrows, 100000)
        steps = df.shape[0] // step_size
        for step in range(steps):
            idx = step*step_size
            if step == steps:
                dftxs = df.iloc[idx:]
                db.bulk_insert_mappings(cls, dftxs.to_dict(orient='records'))
                db.commit()
            else:
                dftxs = df.iloc[idx:idx+step_size]
                db.bulk_insert_mappings(cls, dftxs.to_dict(orient='records'))
                db.commit()
