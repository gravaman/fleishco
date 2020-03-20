import argparse
import datetime
import threading
import pandas as pd
from pandas.tseries.offsets import BDay
from ibapi import wrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.utils import iswrapper


class TestClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)


class TestWrapper(wrapper.EWrapper):
    def __init__(self):
        wrapper.EWrapper.__init__(self)


class DataHandler(TestWrapper, TestClient):
    def __init__(self):
        self.filepath = None
        self.data_type = None
        TestWrapper.__init__(self)
        TestClient.__init__(self, wrapper=self)

    def start(self):
        ticker = "AINC"
        tstr = ticker.lower()
        contract = Contract()
        contract.symbol = ticker
        contract.secType = "STK"
        contract.currency = "USD"
        contract.exchange = "SMART"
        # contract.primaryExchange = "AMEX"

        cur_dt = pd.datetime.today()
        cur_dtstr = cur_dt.strftime('%Y_%m_%d')
        fp = f'data/{tstr}/{tstr}_rebate_data_{cur_dtstr}.csv'

        end_dt = cur_dt

        i = 0
        days = 1
        while True:
            if i >= days:
                break

            end_dtstr = end_dt.strftime('%Y%m%d %H:%M:%S')
            t = threading.Timer(10.0*i, self.make_req,
                                args=(contract, fp, end_dtstr))
            t.start()
            end_dt = end_dt - BDay(1)
            i += 1

    def make_req(self, contract, fp, end_dtstr):
        print(f'Requesting trade data for: {end_dtstr}')
        self.historicalData_req(contract,
                                filepath=fp,
                                end_dt=end_dtstr,
                                duration='1 D',
                                period='1 day',
                                data_type='TRADES')

    @iswrapper
    def historicalData(self, req_id, bar):
        print(bar)
        data = []
        if self.data_type == 'TRADES':
            data = [bar.date, bar.open, bar.high, bar.low, bar.close,
                    bar.volume, bar.average]
        elif self.data_type == 'REBATE_RATE':
            data = [bar.date, bar.open, bar.high, bar.low, bar.close]
        elif self.data_type == 'BID_ASK':
            data = [bar.date, bar.open, bar.high, bar.low, bar.close]
        elif self.data_type == 'HISTORICAL_VOLATILITY':
            data = [bar.date, bar.open]
        else:
            raise Exception('data type has not been properly set')

        print(f'cleaned data: {data}')
        # with open(self.filepath, 'a+') as f:
        #     w = csv.writer(f)
        #     w.writerow(data)

    def historicalData_req(self, contract, filepath=None, end_dt=None,
                           duration='3 D', period='1 day', data_type='TRADES'):
        print('received req for filepath:', filepath)
        if data_type == 'TRADES':
            self.data_type = 'TRADES'
        elif data_type == 'REBATE_RATE':
            self.data_type = 'REBATE_RATE'
        elif data_type == 'HISTORICAL_VOLATILITY':
            self.data_type = 'HISTORICAL_VOLATILITY'
        elif data_type == 'BID_ASK':
            self.data_type = 'BID_ASK'
        else:
            raise Exception('invalid data type provided')

        if filepath is None:
            filepath = contract.symbol.lower() + '_data.csv'

        self.filepath = filepath

        if end_dt is None:
            end_dt = datetime.datetime.now().strftime('%Y%m%d %H:%M:%S')

        self.reqHistoricalData(18001, contract, end_dt, duration,
                               period, data_type, 1, 1, False, [])


def main():
    cmdLineParser = argparse.ArgumentParser("api tests")
    cmdLineParser.add_argument("-p", "--port", action="store", type=int,
                               dest="port", default=7496, help="TCP port")
    args = cmdLineParser.parse_args()

    app = DataHandler()
    app.connect("127.0.0.1", args.port, clientId=0)

    print(f'connected to ibkr on port: {args.port}')
    print(f'serverVersion: {app.serverVersion()}')
    print(f'connectionTime: {app.twsConnectionTime()}\n')

    app.start()
    app.run()


if __name__ == "__main__":
    main()
