import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Date, Boolean
from DB import Base, db


class Corporate(Base):
    __tablename__ = 'corporate'
    id = Column(Integer, primary_key=True)
    finra_symbol = Column(String(14))
    cusip_id = Column(String(9))
    bsym_id = Column(String(12))
    sub_prdct_type = Column(String(10))
    debt_type_cd = Column(String(8))
    issuer_nm = Column(String(80))
    scrty_ds = Column(String(80))
    cpn_rt = Column(Float)
    cpn_type_cd = Column(String(10))
    trd_rpt_efctv_dt = Column(Date)
    mtrty_dt = Column(Date)
    grade = Column(String(1))
    ind_144a = Column(Boolean)

    @classmethod
    def insert_corporates(cls, corps_path):
        # load data
        df = pd.read_csv(corps_path,
                         parse_dates=['trd_rpt_efctv_dt', 'mtrty_dt'])
        cmap = {'bond_sym_id': 'finra_symbol'}
        df = df.rename(columns=cmap)

        # drop duplicate records and records missing required fields
        df = df.drop_duplicates(subset=['finra_symbol'])
        df = df.drop_duplicates(subset=['cusip_id'])
        df = df.drop_duplicates(subset=['bsym_id'])
        df = df.dropna(subset=['finra_symbol', 'cusip_id',
                               'bsym_id', 'cpn_rt', 'cpn_type_cd'])

        # remove converts
        df = df[df.cnvrb_fl != 'Y']

        # remove unused fields, convert NaN to None, convert booleans
        df = df.drop(labels=['cnvrb_fl', 'company_symbol', 'dissem'], axis=1)
        debt_type_cd = df.debt_type_cd.where(df.debt_type_cd.notnull(), None)
        df.loc[:, 'debt_type_cd'] = debt_type_cd
        df.loc[:, 'ind_144a'] = df.ind_144a == 'Y'

        # insert cleaned data into db table
        db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
        db.commit()


if __name__ == '__main__':
    # create table
    if False:
        Base.metadata.create_all()

    # insert corporate notes data
    if False:
        Corporate.insert_corporates('data/bonds/master_file.csv')
