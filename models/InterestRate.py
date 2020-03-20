from os.path import join
import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Date
from DB import Base, db
from utils import get_tickers


class InterestRate(Base):
    __tablename__ = 'interest_rate'
    id = Column(Integer, primary_key=True)
    ticker = Column(String(30))
    date = Column(Date)
    rate = Column(Float)

    @classmethod
    def insert_interest_rates(cls, rates_dir):
        tickers = get_tickers(rates_dir)
        rate_paths = [join(rates_dir, f'{t}.csv') for t in tickers]
        cmap = {'interest_rate': 'rate'}
        for ticker, rate_path in zip(tickers, rate_paths):
            df = pd.read_csv(rate_path, na_values=['.']).dropna()
            if df.shape[0] > 0:
                df = df.rename(columns=cmap)
                df['ticker'] = ticker
                db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
                db.commit()


if __name__ == '__main__':
    # create table
    if False:
        Base.metadata.create_all()

    # insert interest rate data
    if False:
        InterestRate.insert_interest_rates('data/rates/baml')
