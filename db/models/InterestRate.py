from os import listdir
from os.path import join, isfile
import pandas as pd
from sqlalchemy import Column, Integer, Float, Date
from db.models.DB import Base, db


class InterestRate(Base):
    """
        Interest Rate Data
    """
    __tablename__ = 'interest_rate'
    id = Column(Integer, primary_key=True)
    date = Column(Date, index=True)
    BAMLC0A1CAAASYTW = Column(Float)
    BAMLC0A2CAASYTW = Column(Float)
    BAMLC0A3CASYTW = Column(Float)
    BAMLC0A4CBBBSYTW = Column(Float)
    BAMLH0A1HYBBSYTW = Column(Float)
    BAMLH0A2HYBSYTW = Column(Float)
    BAMLH0A3HYCSYTW = Column(Float)
    BAMLC1A0C13YSYTW = Column(Float)
    BAMLC2A0C35YSYTW = Column(Float)
    BAMLC3A0C57YSYTW = Column(Float)
    BAMLC4A0C710YSYTW = Column(Float)
    BAMLC7A0C1015YSYTW = Column(Float)
    BAMLC8A0C15PYSYTW = Column(Float)

    @classmethod
    def insert_interest_rates(cls, rdir):
        pends = [p for p in listdir(rdir) if isfile(join(rdir, p))]
        targets = [(p.split('.csv')[0], join(rdir, p)) for p in pends]
        df = None
        for field, rates_path in targets:
            dftmp = pd.read_csv(rates_path, na_values=['.']).dropna()
            if dftmp.shape[0] > 0:
                dftmp = dftmp.rename(columns={'interest_rate': field}) \
                            .set_index('date')
                if df is None:
                    df = dftmp.copy()
                else:
                    df = df.join(dftmp, how='outer')
        df = df.reset_index().dropna()
        db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
        db.commit()
