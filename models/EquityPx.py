from os.path import join
import pandas as pd
from sqlalchemy import Column, Integer, BigInteger, Float, String, Date
from models.DB import Base, db
from models.utils import get_tickers


class EquityPx(Base):
    __tablename__ = 'equity_px'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10))
    date = Column(Date)
    volume = Column(BigInteger)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)

    @classmethod
    def insert_equity_pxs(cls, equities_dir):
        tickers = get_tickers(equities_dir)
        pxpaths = [join(equities_dir, f'{t}.csv') for t in tickers]
        cmap = {
            'Date': 'date', 'Volume': 'volume',
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Adj Close': 'adj_close'
        }
        for ticker, pxpath in zip(tickers, pxpaths):
            df = pd.read_csv(pxpath).dropna()
            if df.shape[0] > 0:
                df = df.rename(columns=cmap)
                df['ticker'] = ticker
                db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
                db.commit()


if __name__ == '__main__':
    # create table
    if False:
        Base.metadata.create_all()

    # insert equity price data
    if False:
        EquityPx.insert_equity_pxs('data/equities')
