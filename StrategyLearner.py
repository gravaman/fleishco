import os
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import indicators as indi
import marketsimcode as msim
from LoanEnv import LoanEnv
from Plotter import Plotter
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines.bench import Monitor
from stable_baselines import DQN
from stable_baselines import results_plotter


class StrategyLearner:
    def __init__(self, metrics=[indi.pct_sma, indi.rsi],
                 standards=[True, True], ws=[[20], [5]], log_dir='tmp/'):
        # set training params
        self.metrics = metrics
        self.standards = standards
        self.ws = ws

        # set logging directory
        if log_dir:
            self.log_dir = log_dir
            os.makedirs(self.log_dir, exist_ok=True)

        # n_steps used for callback debugging
        self.n_steps = 0

    def train(self, symbol='JPM', sd=dt.datetime(2009, 1, 1),
              ed=dt.datetime(2010, 12, 31), time_steps=int(1e5),
              savepath=None, should_plot=False):
        # load data and indicators
        df = self._load_data([symbol], sd, ed)
        df_met = self._get_indicators(symbol, df)

        # set environment
        self.env = Monitor(LoanEnv(df_met), self.log_dir,
                           allow_early_resets=True)

        # train model
        self.model = DQN(MlpPolicy, self.env, prioritized_replay=True,
                         verbose=1)
        self.model.learn(total_timesteps=time_steps, callback=self.debugcb)

        # save and plot
        if savepath is not None:
            self.model.save(savepath)

        if should_plot:
            results_plotter.plot_results([self.log_dir], time_steps,
                                         results_plotter.X_TIMESTEPS,
                                         f'DQN {symbol}')
            plt.show()

    def cmp_policy(self, symbol='JPM', sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 12, 31), sv=1e5, notional=1e3,
                   commission=0.0, impact=0.0, should_show=False,
                   should_save=False, save_path=None):
        df_trades = self.test_policy(symbol=symbol, sd=sd, ed=ed, sv=sv,
                                     notional=notional)
        sp = msim.compute_portvals(df_trades, start_val=sv,
                                   commission=commission, impact=impact)
        bp = self.benchmark_policy(symbol, sd=sd, ed=ed, sv=sv,
                                   notional=notional, commission=commission,
                                   impact=impact)
        df_cmp = pd.concat([bp, sp], axis=1)
        labels = ['benchmark', 'learner']
        df_cmp.columns = labels
        df_cmp.benchmark /= bp.iloc[0]
        df_cmp.learner /= sp.iloc[0]

        if should_show:
            pltr = Plotter()
            title = f'{symbol} Strategy'
            yax_label = 'Indexed MV'
            X = np.array([df_cmp.index for _ in labels])
            Y = df_cmp.values.T
            colors = [(1, 0, 0), (0, 1, 0)]
            pltr.plot(X, Y, labels=labels, yax_label=yax_label,
                      title=title, colors=colors, should_show=should_show,
                      should_save=should_save, save_path=save_path)

        return df_cmp

    def test_policy(self, symbol='JPM', sd=dt.datetime(2009, 1, 1),
                    ed=dt.datetime(2010, 12, 31), sv=1e5, notional=1e3):
        """
        Tests existing policy against new data
        """
        # load data and indicators
        df = self._load_data([symbol], sd, ed)
        df_met = self._get_indicators(symbol, df)
        df_trades = pd.DataFrame(index=df_met.Date)
        df_trades['Shares'] = 0

        positions = np.zeros((df_trades.shape[0],))

        # new env for testing
        env = self.model.get_env()
        obs = env.reset()

        # initial state and action
        action, _states = self.model.predict(obs)
        positions[0] = np.clip(action, -1, 1)
        obs, rewards, done, info = env.step(action)

        # pass remaining samples thru policy
        i = 1
        while True:
            action, _states = self.model.predict(obs)
            if action == LoanEnv.BUY:
                positions[i] = np.clip(positions[i-1]+1, -1, 1)
            elif action == LoanEnv.SELL:
                positions[i] = np.clip(positions[i-1]-1, -1, 1)
            else:
                raise ValueError(f'unknown action: {action}')
            obs, rewards, done, info = env.step(action)
            if done:
                break
            i += 1

        df_actions = pd.DataFrame(positions, index=df_trades.index,
                                  columns=['Shares'])
        df_actions = df_actions.diff().fillna(positions[0])
        df_trades.update(df_actions)
        df_trades *= notional
        return df_trades.rename(columns={'Shares': symbol})

    def benchmark_policy(self, symbol, sd, ed, sv, notional,
                         commission, impact):
        # load dates and compute buy and hold portvals
        dates = self._load_data(['SPY'], sd, ed).index.get_level_values(1)
        amnts = np.zeros(dates.shape)
        amnts[0] = notional
        df_trades = pd.DataFrame(amnts, index=dates, columns=[symbol])
        vals = msim.compute_portvals(df_trades, start_val=sv,
                                     commission=commission, impact=impact)
        return vals.rename(symbol)

    def debugcb(self, _locals, _globals):
        self.n_steps += 1

    def _load_data(self, symbols, sd, ed):
        return indi.load_data(symbols, pd.date_range(sd, ed))

    def _get_indicators(self, symbol, df):
        df_pxs = pd.DataFrame(df.AdjClose)
        dinps = [df_pxs for _ in self.metrics]
        df_met = df_pxs.copy()
        for i, d, s, w in zip(self.metrics, dinps, self.standards, self.ws):
            df_met = df_met.join(i(d, window_sizes=w, standard=s), how='inner')
        df_met = df_met.loc[symbol].dropna().reset_index()
        return df_met


if __name__ == '__main__':
    lrnr = StrategyLearner()
    tsteps = int(1e4)
    sd = dt.datetime(2009, 1, 1)
    ed = dt.datetime(2011, 12, 31)
    lrnr.train(symbol='JPM', time_steps=tsteps, sd=sd, ed=ed,
               savepath='deepq_loanenv')
    lrnr.cmp_policy(symbol='JPM', sd=sd, ed=ed, sv=1e5, notional=1e3,
                    commission=0.0, impact=0.0, should_show=True)
