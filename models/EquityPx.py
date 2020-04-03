from os.path import join
import pandas as pd
from sqlalchemy import (
    Column, Integer, BigInteger,
    Float, String, Date, ForeignKey
)
from models.DB import Base, db
from models.utils import get_tickers


class EquityPx(Base):
    __tablename__ = 'equity_px'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), index=True)
    entity_id = Column(Integer,
                       ForeignKey('entity.id'),
                       index=True)
    date = Column(Date, index=True)
    volume = Column(BigInteger)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)

    @classmethod
    def insert_equity_pxs(cls, equities_dir, nrows=None):
        tickers = get_tickers(equities_dir)
        pxpaths = [join(equities_dir, f'{t}.csv') for t in tickers]
        cmap = {
            'Date': 'date', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close'
        }
        for ticker, pxpath in zip(tickers, pxpaths):
            df = pd.read_csv(pxpath, nrows=nrows).dropna()
            if df.shape[0] > 0:
                df = df.rename(columns=cmap)
                df['ticker'] = ticker
                db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
                db.commit()
