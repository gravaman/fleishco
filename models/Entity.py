import pandas as pd
from sqlalchemy import Column, Integer, String
from DB import Base, db


class Entity(Base):
    __tablename__ = 'entity'
    id = Column(Integer, primary_key=True)
    cik = Column(String(10))
    name = Column(String(120))
    cusip6 = Column(String(6))

    @classmethod
    def insert_entities(cls, entity_path):
        # load raw entity data
        target_cols = ['CIK', 'SEC_Name', 'CUSIP6']
        dtypes = {'CIK': str, 'SEC_Name': str, 'CUSIP6': str}
        colnames = {'CIK': 'cik', 'SEC_Name': 'name', 'CUSIP6': 'cusip6'}
        df = pd.read_csv(entity_path,
                         usecols=target_cols,
                         dtype=dtypes)

        # drop duplicate issuers and check for valid cusips
        df = df.rename(columns=colnames).drop_duplicates()
        df = df[df.cusip6.str.contains('^[A-Z0-9]{6}$')]

        # insert data into db
        db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
        db.commit()


if __name__ == '__main__':
    # create table
    if False:
        Base.metadata.create_all()

    # insert data
    if False:
        Entity.insert_entities('data/ciks/CIK_CUSIP.csv')
