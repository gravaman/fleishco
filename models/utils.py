from os import listdir
from os.path import isfile, join


def get_tickers(tdir, ftype='.csv'):
    ts = [f.split(ftype)[0] for f in listdir(tdir) if isfile(join(tdir, f))]
    ts = [t.upper() for t in ts]
    return ts
