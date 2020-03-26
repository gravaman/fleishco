import pandas as pd
from sqlalchemy import Column, Integer, Float, String, Date
from models.DB import Base, db


class InterestRate(Base):
    __tablename__ = 'interest_rate'
    id = Column(Integer, primary_key=True)
    date = Column(Date)
    rate = Column(Float)
    ticker = Column(String(20))

    __mapper_args__ = {
        'polymorphic_on': ticker,
        'polymorphic_identity': 'interest_rate'
    }

    @classmethod
    def insert_interest_rates(cls, ticker, rates_path, nrows=None):
        cmap = {'interest_rate': 'rate'}
        df = pd.read_csv(rates_path,
                         na_values=['.'],
                         nrows=nrows).dropna()
        if df.shape[0] > 0:
            df = df.rename(columns=cmap)
            df['ticker'] = ticker
            db.bulk_insert_mappings(cls, df.to_dict(orient='records'))
            db.commit()


class BamlAAASYTW(InterestRate):
    """
    ICE BofA AAA US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_aaa_sytw'
    }


class BamlAASYTW(InterestRate):
    """
    ICE BofA AA US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_aa_sytw'
    }


class BamlASYTW(InterestRate):
    """
    ICE BofA A US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_a_sytw'
    }


class BamlBBBSYTW(InterestRate):
    """
    ICE BofA BBB US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_bbb_sytw'
    }


class BamlBBSYTW(InterestRate):
    """
    ICE BofA BB US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_bb_sytw'
    }


class BamlBSYTW(InterestRate):
    """
    ICE BofA B US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_b_sytw'
    }


class BamlCSYTW(InterestRate):
    """
    ICE BofA C US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_c_sytw'
    }


class Baml13YSTW(InterestRate):
    """
    ICE BofA 1-3 Year US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_13_sytw'
    }


class Baml35YSTW(InterestRate):
    """
    ICE BofA 3-5 Year US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_35_sytw'
    }


class Baml57YSTW(InterestRate):
    """
    ICE BofA 5-7 Year US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_57_sytw'
    }


class Baml710YSTW(InterestRate):
    """
    ICE BofA 7-10 Year US Corporate Index Semi-Annual YTW
    """
    __mapper_args__ = {
        'polymorphic_identity': 'baml_710_sytw'
    }
