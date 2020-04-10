from os import environ
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base


# db connection params
db_pass = environ['PG_FLEISHCO_PASS']
DB_URL = f'postgresql+psycopg2://fleishco:{db_pass}@localhost/fleishco'

# create engine and bind to meta
engine = create_engine(DB_URL, echo=True)

# create db session and base
db = sessionmaker(bind=engine)()
Base = declarative_base(bind=engine)
