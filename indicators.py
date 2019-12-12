import os
import pandas as pd


def load_data(symbols, dates, addSPY=True):
    """Read stock data for given symbols from CSV files."""
    df = pd.DataFrame()

    # conditionally add SPY for reference
    if addSPY and 'SPY' not in symbols:
        symbols = ['SPY'] + symbols

    # add data for each symbol provided
    for symbol in symbols:
        df_tmp = pd.read_csv(sym_to_path(symbol), index_col='Date',
                             parse_dates=True, na_values=['nan'])
        mi = pd.MultiIndex.from_product([[symbol], df_tmp.index.values],
                                        names=['Symbol', 'Date'])
        df_tmp = pd.DataFrame(df_tmp.values, index=mi, columns=df_tmp.columns)
        df = pd.concat([df, df_tmp])

    # conditionally filter SPY trading days
    if addSPY or 'SPY' in symbols:
        tdays = df.loc['SPY'].index.values
        df = df.loc[(slice(None), tdays), :]

    # pull available trading days from SPY filtered and fillna
    df = df.loc[(slice(None), dates), :]
    df = df.groupby('Symbol').fillna(method='ffill')
    df = df.groupby('Symbol').fillna(method='bfill')

    # remove whitespace from column names
    df.columns = [c.replace(' ', '') for c in df.columns.values]

    # return data sorted in ascending order
    return df.sort_index()


def sym_to_path(symbol):
    """Return CSV file path given ticker symbol"""
    return os.path.join('data', f'{symbol}.csv')


def pct_sma(df, window_sizes=[5, 10], standard=True):
    tmp = df.copy()
    df_pct_sma = pd.DataFrame(index=tmp.index)
    col_names = tmp.columns.values
    for n in window_sizes:
        pct = tmp/sma(tmp, n)
        pct.columns = [f'{c}_pct_sma_{n}' for c in col_names]
        df_pct_sma = df_pct_sma.join(pct)

    # standardize data across all symbols by feature
    if standard:
        df_pct_sma = (df_pct_sma-df_pct_sma.mean())/df_pct_sma.std()

    return df_pct_sma


def sma(df, n):
    """Simple Moving Average with window size n"""
    return df.reset_index('Symbol').groupby('Symbol').rolling(n).mean()


def vwpc(df, window_sizes=[5, 10], standard=True):
    df_vwpc = pd.DataFrame(index=df.index)
    tmp = pd.DataFrame(df.iloc[:, 0]*df.iloc[:, 1],
                       columns=['weighted'])
    col_names = [df.columns.values[0]]
    for n in window_sizes:
        chg = tmp/tmp.shift(n)-1
        chg.columns = [f'{c}_vwpc_{n}' for c in col_names]
        df_vwpc = df_vwpc.join(chg)

    if standard:
        df_vwpc = (df_vwpc-df_vwpc.mean())/df_vwpc.std()

    return df_vwpc


def rsi(df, window_sizes=[5, 10], standard=False):
    """
        RSI = 100 - 100/1+RS
        RS1 = total_gain/total_loss
        RS2 = [((n-1)total_gain+gain_n]/[(n-1)total_loss+loss_n]
        Note: standard default False given absolute value relevance
    """
    chg = df.copy()
    chg = (chg/chg.shift(1)-1)
    gain = chg[chg >= 0].fillna(0)
    loss = chg[chg < 0].abs().fillna(0)
    gain_grp = gain.reset_index('Symbol').groupby('Symbol')
    loss_grp = loss.reset_index('Symbol').groupby('Symbol')
    df_rsi = pd.DataFrame(index=chg.index)
    col_names = chg.columns.values
    for n in window_sizes:
        tgain = gain_grp.rolling(n).sum()
        tloss = loss_grp.rolling(n).sum()
        rs2 = ((n-1)*tgain+gain)/((n-1)*tloss+loss)
        rsi = 100-100/(1+rs2.fillna(tgain/tloss))
        rsi.columns = [f'{c}_rsi_{n}' for c in col_names]
        df_rsi = df_rsi.join(rsi)

    if standard:
        df_rsi = (df_rsi-df_rsi.mean())/df_rsi.std()

    return df_rsi
