import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Date
from models.DB import Base, db


class CorpTx(Base):
    __tablename__ = 'corp_tx'
    id = Column(Integer, primary_key=True)
    cusip_id = Column(String(9))
    bond_sym_id = Column(String(14))
    company_symbol = Column(String(20))
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
    def insert_corp_txs(cls, txs_path):
        df = pd.read_csv(txs_path).dropna()
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


if __name__ == '__main__':
    # create table
    if False:
        Base.metadata.create_all()

    # insert data
    if False:
        CorpTx.insert_corp_txs('data/bonds/clean_bond_close_pxs.csv')
