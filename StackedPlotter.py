import operator
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib import cm
from matplotlib.ticker import LinearLocator, StrMethodFormatter
from matplotlib.gridspec import GridSpec


class StackedPlotter:
    def __init__(self, stacked_hratios=[2, 1], line_alpha=0.7, line_width=1.2,
                 grid_style='dotted', xtickcnt=7, date_fmt='%Y-%m',
                 tick_color='0.25', bgc='0.90', fc='0.60',
                 highlight_alpha=0.4, title_color='0.15', num_fmt='{x:,.2f}'):
        """
        params:
        - stacked_hratios: height ratios of stacked plots
        - line_alpha: alpha of plot lines
        - line_width: width of plot lines
        - grid_style: grid line style; None turns off gridlines
        - xtickcnt: x-axis major tick count
        - date_fmt: str fmt for date formatter used by axes
        - tick_color: tick mark color (grayscale)
        - bgc: background color (default grayscale)
        - fc: frame color (default grayscale)
        - highlight_alpha: highlighted region alpha
        - title_color: title color (grayscale)
        """
        # default stacked plot settings for grid
        if len(stacked_hratios) != 2:
            raise ValueError(f'stacked_hratios param length must be 2')

        self._stacked_shape = (2, 1)  # stacked grid shape
        self.stacked_hratios = stacked_hratios  # shape of stacked plots
        self._hspace = 0.025  # height reserved for space between subplots

        # colors and shapes
        self._cmap = cm.get_cmap('cividis')
        self.line_alpha = line_alpha
        self.highlight_alpha = highlight_alpha
        self.line_width = line_width
        self.line_colors = []  # series line colors

        # ticks and grids
        self.grid_style = grid_style
        self.xtickcnt = 7
        self.date_fmt = mdates.DateFormatter(date_fmt)
        self.num_fmt = StrMethodFormatter(num_fmt)
        self.tick_color = tick_color
        self.bgc = bgc
        self.fc = fc
        self.title_color = title_color
        self.xhline_color = '0.1'

    def table(self, df, show_table=False, save_path=None):
        plt.figure(figsize=(6, 1))

        # hide axes
        # fig.patch.set_visible(False)
        plt.axis('off')
        plt.grid('off')

        # generate table
        plt.table(cellText=df.values, colLabels=df.columns, loc='center',
                  rowLabels=df.index.values, rowLoc='left', cellLoc='center')
        # fig.tight_layout()

        # display and save table
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if show_table:
            plt.show()

        plt.clf()

    def stacked_plot(self, X1, X2, x1_labels=None, x2_labels=None,
                     yax_labels=None, show_top_leg=True, show_bot_leg=False,
                     ycs=None, save_path=None, should_show=False, title=None,
                     colors=None, hcolors=None, yc_data=None):
        """
            X1: pd df with each column a line for top plot
            X2: pd df with each column a line for bottom plot
            yax_labels: list of ylabels; if len 1 no x2 label
            show_top_leg: top plot legend toggle
            show_bot_leg: bottom plot legend toggle
            ycs: list of y constraints [src, [op, threshold, freq]]
                - example: [[1, ['>', 1.2, 4]]]
                - each constraint drawn on ax1 but src may be either ax
            save_path: path to save figures
            should_show: bool toggle for displaying plot
            title: chart title str
            yc_key: used as bottom of constraint drawings
            colors: used for lines
            hcolors: used for constraint lines
        """
        DATA = [X1, X2]
        fig = plt.figure()
        gs = GridSpec(*self._stacked_shape,
                      height_ratios=self.stacked_hratios,
                      hspace=self._hspace)

        # check for y-axis labels and clean
        if yax_labels is None:
            yax_labels = [None, None]

        if len(yax_labels) == 1:
            yax_labels.append(None)

        # outliers areas and highlight colors
        if ycs is not None:
            YC_DATA = [X1, X2]
            if yc_data is not None:
                YC_DATA.append(yc_data)
            outs = [self._outlier_idxs(YC_DATA[i-1], c) for i, c in ycs]
            if hcolors is None:
                hcolors = self._get_colors(len(outs))

        legend_toggles = [show_top_leg, show_bot_leg]

        axes = []
        for i, X in enumerate(DATA):
            ax = fig.add_subplot(gs[i])
            axes.append(ax)
            if colors is None:
                colors = self._get_colors(X.shape[1])
            if i == 0:
                ax.set_title(title, color=self.title_color)

            # add gridlines
            if self.grid_style is not None:
                ax.grid(linestyle=self.grid_style)

            # add ylabel
            if yax_labels[i] is not None:
                ax.set_ylabel(yax_labels[i], color=self.tick_color)

            lcolors = colors[i]
            for j, col in enumerate(X.columns.values):
                x = X[col]
                ax.plot(x.index, x, color=lcolors[j], label=col,
                        alpha=self.line_alpha, linewidth=self.line_width)

                ax.xaxis.set_major_locator(LinearLocator(self.xtickcnt))
                ax.xaxis.set_major_formatter(self.date_fmt)
                ax.yaxis.set_major_formatter(self.num_fmt)
                ax.tick_params(colors=self.tick_color)
                ax.set_facecolor(self.bgc)
                plt.setp(ax.spines.values(), color=self.fc)

            # highlight outliers
            if ycs is not None:
                ylims = ax.get_ylim()
                # highlight each set of indices for constraint outliers
                if i == 0:
                    col = X.columns.values[0]
                    for ycidx, df_out in enumerate(outs):
                        # first long second short
                        h = (ylims[1]-ylims[0])*0.2
                        vals = X[X.index.isin(df_out.index)][col]
                        if ycidx == 1:
                            vals = vals-h
                        ax.bar(x=df_out.index, height=h,
                               bottom=vals, width=self.line_width,
                               color=hcolors[ycidx],
                               alpha=self.highlight_alpha)
                else:
                    # draw dashed threshold
                    for _, constraint in ycs:
                        ax.axhline(constraint[1], color=self.xhline_color,
                                   alpha=self.line_alpha, ls=self.grid_style)

            # add legend
            if legend_toggles[i]:
                leg = ax.legend()
                for txt in leg.get_texts():
                    txt.set_color(self.tick_color)
                for leghand in leg.legendHandles:
                    leghand.set_alpha(self.line_alpha)

        # align axes and hide top plot x-axis
        fig.align_ylabels(axes)
        axes[0].set_xlim(axes[1].get_xlim())
        axes[0].tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)

        # storage and display
        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if should_show:
            plt.show()

        plt.clf()

    def _outlier_idxs(self, X, constraint):
        """
            Finds indices of areas outside given constraints

            params:
            - X: pd series to compare
            - constraint: [operator str, threshold val, min size]
            returns:
            - indices meeting constraint
        """
        op, threshold, min_size = constraint
        op = self._get_operator(op)
        bounds = op(X, threshold)
        mask = bounds
        for i in range(min_size):
            mask = mask & bounds.shift(i)
        mask = mask.fillna(False)
        return X[mask].dropna()

    def _get_colors(self, count, scheme=None):
        if scheme is None:
            scheme = self._cmap
        return [scheme(r) for r in np.random.ranf(size=count)]

    def _get_operator(self, op):
        return {
            '>': operator.gt,
            '>=': operator.ge,
            '<': operator.lt,
            '<=': operator.le,
            '=': operator.eq,
        }[op]
