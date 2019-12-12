import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, StrMethodFormatter
import matplotlib.dates as mdates
from matplotlib import cm


class Plotter:
    def __init__(self, line_alpha=0.7, line_width=1.2, tick_color='0.25',
                 bgc='0.90', fc='0.60', title_color='0.15',
                 num_fmt='{x:,.2f}', grid_style=None,
                 xtickcnt=7, date_fmt='%Y-%m'):
        self.line_alpha = line_alpha
        self.line_width = line_width
        self.tick_color = tick_color
        self.bgc = bgc
        self.fc = fc
        self.title_color = title_color
        self.date_fmt = mdates.DateFormatter(date_fmt)
        self.num_fmt = StrMethodFormatter(num_fmt)
        self.grid_style = grid_style
        self.xtickcnt = xtickcnt
        self._cmap = cm.get_cmap('cividis')

    def plot(self, X, Y, labels=None, xax_label=None, yax_label=None,
             colors=None, title=None, should_show=False, should_save=False,
             save_path=None):

        if labels is None:
            labels = [f'plot {i+1}' for i in range(Y.shape[0])]

        if colors is None:
            colors = self._get_colors(X.shape[0])

        fig, ax = plt.subplots()

        for x, y, label, c in zip(X, Y, labels, colors):
            ax.plot(x, y, label=label, alpha=self.line_alpha,
                    linewidth=self.line_width, color=c)

        if title:
            ax.set_title(title, color=self.title_color)

        if self.grid_style:
            ax.grid(linestyle=self.grid_style)

        if xax_label:
            ax.set_xlabel(xax_label, color=self.tick_color)

        if yax_label:
            ax.set_ylabel(yax_label, color=self.tick_color)

        ax.xaxis.set_major_locator(LinearLocator(self.xtickcnt))
        ax.xaxis.set_major_formatter(self.date_fmt)
        ax.yaxis.set_major_formatter(self.num_fmt)
        ax.tick_params(colors=self.tick_color)
        ax.set_facecolor(self.bgc)
        plt.setp(ax.spines.values(), color=self.fc)

        leg = ax.legend()
        for txt in leg.get_texts():
            txt.set_color(self.tick_color)

        for leghand in leg.legendHandles:
            leghand.set_alpha(self.line_alpha)

        if should_save:
            plt.savefig(save_path, bbox_inches='tight')

        if should_show:
            plt.show()

        plt.clf()

    def _get_colors(self, count, scheme=None):
        if scheme is None:
            scheme = self._cmap
        return [scheme(r) for r in np.random.ranf(size=count)]
