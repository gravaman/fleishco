import yfinance as yf


def pull(ticker, period='max', should_save=False, savepath=None):
    stock = yf.Ticker(ticker.upper())
    dfhist = stock.history(period=period)
    dfhist = dfhist.drop(['Dividends', 'Stock Splits'], axis=1)
    dfhist['Adj Close'] = dfhist.Close
    if should_save:
        if savepath is None:
            savepath = f'data/{ticker.upper()}.csv'
        dfhist.to_csv(savepath)
    return dfhist
