import pandas as pd
import datetime as dt
import indicators as indi


def order_sign(row):
    if row['Order'] == 'SELL':
        row['Shares'] = -row['Shares']
        return row
    else:
        return row


def update_start(groupdf):
    groupdf.ShareChg.iloc[0] = groupdf.iloc[0]['Shares']
    return groupdf


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000,
                     commission=9.95, impact=0.005):
    # this is the function the autograder will call to test your code
    # orders_file may be a string, or it may be a file object
    if isinstance(orders_file, pd.DataFrame):
        orders = orders_file.copy()
    else:
        orders = pd.read_csv(orders_file, index_col=['Date'], parse_dates=True,
                             na_values=['nan']).sort_index()

    symbols = list(orders.columns.values)

    # add trade count column for aggregation purposes
    orders['Symbol'] = symbols[0]
    for symbol in symbols:
        orders = orders.rename(columns={symbol: 'Shares'})

    orders['Trades'] = 1

    # get date range and symbols for indices
    start_date = pd.to_datetime(orders.index.values[0]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(orders.index.values[-1]).strftime('%Y-%m-%d')
    dates = pd.date_range(start_date, end_date)

    # cvt sells to negatives and drop Order col
    if 'Order' in orders.columns.values:
        orders = orders.apply(order_sign, axis=1).drop(['Order'], axis=1)

    # get price data and remove SPY
    pxs = indi.load_data(symbols, dates).AdjClose.drop(['SPY'])
    pxs = pxs.reset_index().set_index(['Symbol', 'Date'])
    pxs = pd.DataFrame(pxs)
    pxs = pxs.rename(columns={'AdjClose': 'Price'})

    # merge orders and prices
    orders = orders.reset_index().set_index(['Symbol', 'Date'])
    portvals = pxs.join(orders)
    portvals = portvals.reset_index(['Symbol'])
    portvals['Price'] = portvals['Price'].fillna(method='ffill')

    # cumulative shares
    portvals['Shares'] = portvals.groupby('Symbol')['Shares'].cumsum()
    portvals['Shares'] = portvals.groupby('Symbol')['Shares'] \
        .fillna(method='ffill').fillna(0)

    # Fill na trades
    portvals['Trades'] = portvals['Trades'].fillna(0)

    # set MV, ShareChg cols; init Commis col
    portvals['MV'] = portvals.Shares*portvals.Price
    portvals['ShareChg'] = portvals.groupby('Symbol')['Shares'].diff()
    portvals[portvals.ShareChg == 0].Trades = 0
    portvals['Commis'] = (-portvals.Trades*commission).fillna(0)

    # update na in ShareChg col of first row to be Shares
    portvals = portvals.reset_index().set_index(['Symbol', 'Date'])
    portvals = portvals.groupby('Symbol').apply(lambda gdf: update_start(gdf))
    portvals = portvals.sort_index()

    # trading impact on mkt
    portvals['Impact'] = (portvals.ShareChg/portvals.ShareChg.abs()) \
        .fillna(0)*impact+1

    # cost basis pf for trading impact
    portvals['Basis'] = -portvals.ShareChg*portvals.Price*portvals.Impact

    # consolidate multiple trades in a given day
    consol = ['Trades', 'Basis', 'ShareChg', 'Commis']
    portvals[consol] = portvals.groupby(['Symbol', 'Date'])[consol].sum(axis=0)
    portvals = portvals.reset_index() \
        .drop_duplicates(subset=['Symbol', 'Date'], keep='last') \
        .set_index(['Symbol', 'Date'])

    # add cash balance
    cashdf = portvals.loc[portvals.index.values[0][0]].copy()
    cashdf = pd.concat([cashdf], keys=['CASH'], names=['Symbol'])
    cashdf.Price = 1.0
    cashdf.Shares = start_val
    cashdf.Trades = 0
    cashdf.ShareChg = 0
    cashdf.MV = cashdf.Price*cashdf.Shares
    cashdf.Commis = 0.0
    portvals = pd.concat([portvals, cashdf])

    # update na in ShareChg col of first row to be Shares
    portvals = portvals.groupby('Symbol').apply(lambda gdf: update_start(gdf))
    portvals = portvals.sort_index()

    # calculate the trade basis
    portvals.loc['CASH'].Impact = 1.0
    nc = portvals.query('Symbol != "CASH"')[['Basis', 'Commis']]
    portvals.loc['CASH'].MV = nc.groupby('Date').sum(axis=1).sum(axis=1)
    portvals.loc['CASH', 'MV'].iloc[0] += start_val
    portvals.loc['CASH'].MV = portvals.loc['CASH'].MV.cumsum()

    # return the mv
    return portvals.MV.groupby('Date').sum(axis=1)


def test_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters
    of = "./orders/orders-01.csv"
    sv = 1000000
    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv,
                                commission=0.0, impact=0.000)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"
    # Get portfolio stats
    # Here we fake data. use code from previous assignments
    start_date = dt.datetime(2008, 1, 1)
    end_date = dt.datetime(2008, 6, 1)
    cr, adr, stdr, sr = [0.2, 0.01, 0.02, 1.5]
    cr_SPY, adr_SPY, stdr_SPY, sr_SPY = [0.2, 0.01, 0.02, 1.5]
    # Compare portfolio against $SPX
    print(f"Date Range: {start_date} to {end_date}")
    print()
    print(f"Sharpe Ratio of Fund: {sr}")
    print(f"Sharpe Ratio of SPY : {sr_SPY}")
    print()
    print(f"Cumulative Return of Fund: {cr}")
    print(f"Cumulative Return of SPY : {cr_SPY}")
    print()
    print(f"Standard Deviation of Fund: {stdr}")
    print(f"Standard Deviation of SPY : {stdr_SPY}")
    print()
    print(f"Average Daily Return of Fund: {adr}")
    print(f"Average Daily Return of SPY : {adr_SPY}")
    print()
    print(f"Final Portfolio Value: {portvals[-1]}")


if __name__ == "__main__":
    test_code()
