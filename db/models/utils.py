from os import listdir
from os.path import isfile, join
import string


# cusip constants
cusipchars = string.digits+string.ascii_uppercase+"*@#"
cusip2value = dict((ch, n) for n, ch in enumerate(cusipchars))
cusipweights = [1, 2, 1, 2, 1, 2, 1, 2]


def to_cusip9(cusip8):
    # converts cusip 8 to cusip 9 by appending checksum value
    return cusip8+checksum(cusip8)


def checksum(cusip8):
    # generates checksum value for given cusip 8
    digits = [(w*cusip2value[ch]) for w, ch in zip(cusipweights, cusip8)]
    cs = sum([(x % 10 + x // 10) for x in digits]) % 10
    cs = str((10 - cs) % 10)
    return cs


def get_tickers(tdir, ftype='.csv'):
    ts = [f.split(ftype)[0] for f in listdir(tdir) if isfile(join(tdir, f))]
    ts = [t.upper() for t in ts]
    return ts
