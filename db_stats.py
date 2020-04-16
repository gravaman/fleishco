from os.path import join
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from db.db_query import (
    get_corptx_ids,
    counts_by_sym,
    get_target_stats
)


TECH = [
    'AAPL', 'MSFT', 'INTC', 'IBM', 'QCOM', 'ORCL', 'TXN', 'MU', 'AMZN', 'GOOG',
    'NVDA', 'JNPR', 'ADI', 'ADBE', 'STX', 'AVT', 'ARW', 'KLAC', 'A', 'NTAP',
    'VRSK', 'TECD', 'KEYS', 'CSCO', 'AMD', 'CRM'
]

LEISURE = ['FUN', 'RCL', 'EPR']
RETAIL = [
    'KSS', 'COST', 'MAT', 'ORLY', 'DG', 'HD', 'BBY', 'GPS', 'RL',
    'TIF', 'ROST', 'BBBY', 'HAS', 'DDS', 'WMT',
    'KR', 'AZO', 'WHR', 'AAP'
]
RESTAURANTS = ['SBUX', 'MCD', 'DRI']
CONSUMER = LEISURE + RETAIL + RESTAURANTS

LODGING = ['H']
HOMEBUILDERS = ['LEN', 'TOL', 'KBH', 'PHM', 'BZH', 'MDC']
SHOPPING_CENTER_REITS = ['REG', 'KIM']
DATA_CENTER_REITS = ['DLR', 'AMT']
TRIPLE_NET_REITS = ['O', 'SRC']
REAL_ESTATE = LODGING + HOMEBUILDERS + SHOPPING_CENTER_REITS + \
    DATA_CENTER_REITS + TRIPLE_NET_REITS

TEST_TECH = ['TECD', 'KEYS']
TEST_RE = ['EPR', 'RCL']
OUTPUT_DIR = 'output/data_metrics'


def summarize_data(sector_tickers, sector_names, tick_limit, ed, periods=[],
                   should_plot=False, savedir=None, data_tag='untitled'):
    train_periods, val_periods, test_periods = periods
    sector_cnts, sector_stats = [], []
    for tickers in sector_tickers:
        df_cnts, df_stats = get_data_metrics(tickers, tick_limit,
                                             ed=ed, periods=sum(periods))
        sector_cnts.append(df_cnts)
        sector_stats.append(df_stats)

    # calculate tx totals by sector and overall
    df_cnt_totals = pd.DataFrame(0, index=sector_cnts[0].index,
                                 columns=sector_names)
    for name, df_cnt in zip(sector_names, sector_cnts):
        df_cnt_totals[name] = df_cnt.sum(axis=1)

    df_cnt_totals['total'] = df_cnt_totals.sum(axis=1)

    # calculate overall statistics
    df_total_stats = pd.DataFrame(0, index=df_cnt_totals.index,
                                  columns=sector_stats[0].columns)
    # combined mu
    for name, df_stat in zip(sector_names, sector_stats):
        df_total_stats.mu += df_stat.mu*df_cnt_totals[name]

    df_total_stats.mu /= df_cnt_totals.total

    # combined sigma
    for name, df_stat in zip(sector_names, sector_stats):
        df_total_stats.sigma += (
            (((df_stat.sigma**2) *
              df_cnt_totals[name])**0.5 +
             df_stat.mu-df_total_stats.mu)**2 /
            df_cnt_totals.total
        )

    df_total_stats.sigma = df_total_stats.sigma**0.5

    # conditionally plot results
    if savedir is not None:
        tx_cnt_path = join(savedir, f'dataset_txcnts_{data_tag}.png')
        sector_stats_path = join(savedir,
                                 f'dataset_txstats_sector_{data_tag}.png')
        all_stats_path = join(savedir, f'dataset_txstats_all_{data_tag}.png')

    if should_plot or savedir is not None:
        cnt_names = sector_names + ['Combined']
        cnt_total = df_cnt_totals.loc[:, df_cnt_totals.columns != 'total']
        data = sector_cnts + [cnt_total]
        _tx_cnt_plot(data, cnt_names, should_plot=should_plot,
                     savepath=tx_cnt_path)

        tmp_periods = sector_stats[0].index.values
        test_bounds = (tmp_periods[-test_periods-1],
                       tmp_periods[-1])
        val_bounds = (tmp_periods[-val_periods-test_periods-1],
                      test_bounds[1])
        _stats_plot(sector_stats,
                    sector_names,
                    highlight_bounds=[val_bounds, test_bounds],
                    should_plot=should_plot, savepath=sector_stats_path)
        _stats_plot([df_total_stats],
                    ['Combined'],
                    highlight_bounds=[val_bounds, test_bounds],
                    should_plot=should_plot, savepath=all_stats_path)

    # print results to console
    sector_names.append('Combined')
    sector_cnts.append(df_cnt_totals)
    sector_stats.append(df_total_stats)

    for name, cnts, stats in zip(sector_names, sector_cnts, sector_stats):
        # add row for total across all years
        total = pd.DataFrame(cnts.sum(axis=0).values[None, :],
                             index=['Total'], columns=cnts.columns.values)
        cnts = pd.concat([cnts, total])
        print('-'*89)
        print(f'{name} Transaction Counts:\n')
        print(cnts)
        print()
        if savedir is not None:
            p = join(savedir, f'dataset_txcnts_{name.lower()}.csv')
            cnts.to_csv(p)

        print(f'{name} Transaction Pcts:\n')
        if name != 'Combined':
            print(cnts.div(cnts.sum(axis=1), axis=0))
        else:
            print(cnts.div(cnts.total, axis=0))
        print()
        if savedir is not None:
            p = join(savedir, f'dataset_txpcts_{name.lower()}.csv')
            cnts.to_csv(p)

        print(f'{name} Transaction Stats:\n')
        print(stats)
        print('-'*89)
        if savedir is not None:
            p = join(savedir, f'dataset_txstats_{name.lower()}.csv')
            cnts.to_csv(p)


def get_data_metrics(tickers, tick_limit, ed, periods, freq='Y',
                     release_window=720, T=8):
    # build date range for queries
    dts = pd.date_range(end=ed, periods=periods+1, freq=freq)
    dts = dts.map(lambda x: x.strftime('%Y-%m-%d')).values
    sds, eds = dts[:-1], dts[1:]

    # pull data by ticker and year
    df_cnt = pd.DataFrame(np.zeros((periods, len(tickers)), dtype=np.integer),
                          columns=tickers, index=eds)
    df_stat = pd.DataFrame(np.zeros((periods, 2), dtype=np.float),
                           columns=['mu', 'sigma'], index=eds)
    for sd, ed in zip(sds, eds):
        # get ids for given period
        period_ids = get_corptx_ids(tickers,
                                    release_window=release_window,
                                    release_count=T,
                                    limit=None,
                                    tick_limit=tick_limit,
                                    sd=sd, ed=ed).tolist()
        # count ids by ticker
        ticker_counts = counts_by_sym(period_ids)
        for ticker, cnt in ticker_counts:
            df_cnt.loc[ed, ticker] = cnt

        # get target stats for ids
        df_stat.loc[ed] = get_target_stats(period_ids)

    return df_cnt, df_stat


def _tx_cnt_plot(sector_data, sector_names, should_plot=False,
                 savepath=None):
    fig, ax = plt.subplots()

    # plot bars
    margin = 0.1
    bars = len(sector_data)
    width = (1-margin)/bars
    xcount = sector_data[0].shape[0]
    x = np.arange(xcount)
    for i, df in enumerate(sector_data):
        ax.bar(x+i*width, df.sum(axis=1).values, align='edge',
               width=width, alpha=0.4)

    # add dates to x-axis
    date_xlocs = np.linspace(width*bars*0.5,
                             xcount-1+width*bars*0.5,
                             xcount)
    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    ax.set_xticks(date_xlocs)
    ax.set_xticklabels([])

    y_min, y_max = ax.get_ylim()
    date_yloc = y_min-(y_max-y_min)*0.175
    date_labels = [f'FYE {d[:4]}' for d in sector_data[0].index.values]
    for xloc, label in zip(date_xlocs, date_labels):
        ax.text(xloc, date_yloc, label, ha='center', fontsize='x-small')

    # add sector labels to x-axis
    base_xlocs = np.linspace(width*0.5, xcount-1+width*0.5, xcount)
    sector_yloc = y_min-(y_max-y_min)*0.01
    sector_labels = zip(*[sector_names for _ in range(xcount)])
    for i, labels in enumerate(sector_labels):
        sector_xlocs = base_xlocs+i*width
        for xloc, label in zip(sector_xlocs, labels):
            ax.text(xloc, sector_yloc, label, ha='right',
                    va='top', rotation=45, fontsize='xx-small')

    # format y axis numbers
    fmt = '{x:,.0f}'
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter(fmt))

    ax.set_ylabel('Transaction Count')
    ax.set_title('Transactions By Issuer')

    plt.tight_layout()

    if should_plot:
        plt.show()
    if savepath is not None:
        plt.savefig(savepath)


def _stats_plot(sector_data, sector_names, highlight_bounds, should_plot=False,
                savepath=None):
    fig, ax = plt.subplots()

    date_labels = [f'FYE {d[:4]}' for d in sector_data[0].index.values]
    for df, name in zip(sector_data, sector_names):
        ax.plot(date_labels, df['mu'].values, label=name)
        ax.fill_between(date_labels, df['mu']-df['sigma'],
                        df['mu']+df['sigma'], alpha=0.2)

    date_map = {dt: lbl for dt, lbl in
                zip(sector_data[0].index.values, date_labels)}
    val_bounds, test_bounds = highlight_bounds
    val_bounds = [date_map[bound] for bound in val_bounds]
    test_bounds = [date_map[bound] for bound in test_bounds]
    ax.axvspan(*val_bounds, alpha=0.1, color='y')
    ax.axvspan(*test_bounds, alpha=0.1, color='g')

    ax.tick_params(axis='x', labelsize='x-small')

    y_min, y_max = ax.get_ylim()
    y_loc = y_min+(y_max-y_min)/10
    for i, xval in enumerate(sector_data[0].index.values):
        if xval == val_bounds[0]:
            ax.text(i/2, y_loc, 'train', ha='center')
            ax.text(i+0.5, y_loc, 'validation', ha='center')
        elif xval == test_bounds[0]:
            ax.text(i+0.5, y_loc, 'test', ha='center')

    ax.legend(loc='upper left')
    ax.set_ylabel('Yield To Worst ($\mu \pm \sigma$)')  # noqa - latex
    ax.set_title('Target Statistics Over Sample Period')

    plt.tight_layout()

    if should_plot:
        plt.show()
    if savepath is not None:
        plt.savefig(savepath)


def main():
    # sector_tickers = [TEST_TECH, TEST_RE]
    # sector_names = ['Tech', 'Real Estate']
    # periods = [2, 1, 1]
    # tick_limit = 10
    sector_tickers = [TECH, CONSUMER, REAL_ESTATE]
    sector_names = ['Tech', 'Consumer', 'Real Estate']
    periods = [6, 1, 1]
    tick_limit = 1000
    summarize_data(sector_tickers, sector_names, tick_limit=tick_limit,
                   ed='2019-12-31', periods=periods, should_plot=True,
                   savedir=OUTPUT_DIR, data_tag='tech_consumer_re')


if __name__ == '__main__':
    main()
