import numpy as np
import pandas as pd
import datetime as dt
import gym
from gym import spaces
import indicators as indi


class LoanEnv(gym.Env):
    """
    Custom Environment for levered loans that follows gym interface.

    actions:
        - SELL: 0
        - BUY: 1
    """
    metadata = {'render.modes': ['console']}
    SELL = 0
    BUY = 1

    def __init__(self, df, bincnt=0):
        """
        params:
        - df: NxM df with cols: [Date, AdjClose, Sig1, Sig2, ..., SigM]
        - bincnt: bins for discretizing state space; continous if 0
        """
        super().__init__()

        self.df = df.copy()
        self.dates = self.df.Date
        self.pxchg = self.df.AdjClose/self.df.AdjClose.shift(1)-1.0

        # get signals
        self.signals = self.df.drop(['Date', 'AdjClose'], axis=1)

        # optionally discretize based on bincnt
        self.bincnt = bincnt
        if self.bincnt > 0:
            for col in self.signals.columns.values:
                self.signals[col] = pd.cut(self.signals[col].values, bincnt,
                                           labels=False).astype(np.float32)

        # observation_space float32 enabling continuous
        self.observation_space = spaces.Box(low=self.signals.min().values,
                                            high=self.signals.max().values,
                                            dtype=np.float32)
        self.today = 0
        self.last_day = df.shape[0]-1

        # action space consists of SELL and BUY
        self.action_space = spaces.Discrete(2)
        self.action = None

        # define allowable position range [0, pos_ceil]
        self.pos_range = [-1.0, 1.0]

        # agent starts flat
        self.agent_pos = 0.0

    def reset(self):
        """
        gym interface requirement. Starts agent with flat position on first
        day of data period.

        returns:
        - obs: np array with starting signals
        """
        self.agent_pos = 0.0
        self.today = 0
        return self.signals.iloc[self.today].values

    def step(self, action):
        """
        gym interface requirement. Steps forward 1 step in environment

        params:
        - action: SELL (0) or BUY (1)

        return:
        - obs: np array with updated signals
        - reward: pxchg for action taken
        - done: boolean for whether period completed
        - info: optional info
        """
        # update position for action
        self.action = action
        if action == self.SELL:
            self.agent_pos -= 1
        elif action == self.BUY:
            self.agent_pos += 1
        else:
            raise ValueError(f'action must either be SELL or BUY')

        # clip position to allowable range
        self.agent_pos = np.clip(self.agent_pos, *self.pos_range)

        # step forward 1 day
        self.today += 1
        done = self.today >= self.last_day

        # reward with pxchg for position
        reward = self.pxchg[self.today]*self.agent_pos

        # optional info
        info = {'agent_pos': self.agent_pos, 'day': self.today}

        # obs updated market env as defined by signal
        obs = self.signals.iloc[self.today].values

        return obs, reward, done, info

    def render(self, mode='console'):
        """
        gym interface requirement. Renders based on provided mode.
        Currently only console mode implemented.
        """
        if mode != 'console':
            raise NotImplementedError()

        # prints date action position and reward
        today = self.dates.iloc[self.today].strftime('%Y-%m-%d')
        action = 'BUY' if self.action == self.BUY else 'SELL'
        msg = (
            f'{today} '
            f'{action} --> '
            f'position: {self.agent_pos} '
            f'({self.pxchg.iloc[self.today]*100:0.2f}%)'
        )
        print(msg)

    def close(self):
        """
        gym interface requirement
        """
        pass


if __name__ == '__main__':
    print(f'LoanEnv Main Called')
    symbols = ['JPM']
    dates = pd.date_range(dt.datetime(2009, 1, 1), dt.datetime(2010, 12, 31))
    df = indi.load_data(symbols, dates)
    df_pxs = pd.DataFrame(df.AdjClose)
    df_met = df_pxs.copy()

    # generate indicators
    metrics = [indi.pct_sma, indi.rsi]
    dinps = [df_pxs, df_pxs]
    standards = [True, False]
    ws = [[20], [5]]
    for i, d, s, w in zip(metrics, dinps, standards, ws):
        df_met = df_met.join(i(d, window_sizes=w, standard=s), how='inner')

    df_met = df_met.loc['JPM'].reset_index().dropna().reset_index()
    loan_env = LoanEnv(df_met)
    for step in range(5):
        print(f'Step {step+1}')
        obs, reward, done, info = loan_env.step(loan_env.BUY)
        print(f'obs={obs} reward={reward} done={done}')
        loan_env.render()
        if done:
            print(f'Goal reached! reward={reward}')
            break
