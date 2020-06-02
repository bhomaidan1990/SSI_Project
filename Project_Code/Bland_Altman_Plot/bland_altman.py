#=====================================
# Reference : https://github.com/josesho/bland_altman/blob/master/bland_altman.py
#=====================================
import numpy as np
import matplotlib.pyplot as plt

def bland_altman_plot(m1, m2,
                      sd_limit=1.96,
                      ax=None,
                      scatter_kwds=None,
                      mean_line_kwds=None,
                      limit_lines_kwds=None):
    """
    Bland-Altman Plot.

    A Bland-Altman plot is a graphical method to analyze the differences
    between two methods of measurement. The mean of the measures is plotted
    against their difference.

    Parameters
    ----------
    m1, m2: 1D array-like or pandas Series

    sd_limit : float, default 1.96
        The limit of agreements expressed in terms of the standard deviation of
        the differences. If `md` is the mean of the differences, and `sd` is
        the standard deviation of those differences, then the limits of
        agreement that will be plotted will be
                       md - sd_limit * sd, md + sd_limit * sd
        The default of 1.96 will produce 95% confidence intervals for the means
        of the differences.
        If sd_limit = 0, no limits will be plotted, and the ylimit of the plot
        defaults to 3 standard deviatons on either side of the mean.

    ax: matplotlib.axis, optional
        matplotlib axis object to plot on.

    scatter_kwargs: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.scatter plotting method

    mean_line_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

    limit_lines_kwds: keywords
        Options to to style the scatter plot. Accepts any keywords for the
        matplotlib Axes.axhline plotting method

   Returns
    -------
    Plot of Bland Altman
    # ax: matplotlib Axis object
    """
    # Compare the lengths of the samples coming from a two different methods
    if len(m1) != len(m2):
        raise ValueError('m1 does not have the same length as m2.')
    if sd_limit < 0:
        raise ValueError('sd_limit ({}) is less than 0.'.format(sd_limit))
    # Compute the mean of the samples for both of them
    means = np.mean([m1, m2], axis=0)
    # Compute the difference between samples
    diffs = m1 - m2
    # Compute the mean difference between the samples
    mean_diff = np.mean(diffs)
    # Compute the standard deviation of the difference between samples
    std_diff = np.std(diffs, axis=0)

    # Plot Options for nice plot
    if ax is None:
        ax = plt.gca()

    scatter_kwds = scatter_kwds or {}
    if 's' not in scatter_kwds:
        scatter_kwds['s'] = 20
    mean_line_kwds = mean_line_kwds or {}
    limit_lines_kwds = limit_lines_kwds or {}
    for kwds in [mean_line_kwds, limit_lines_kwds]:
        if 'color' not in kwds:
            kwds['color'] = 'gray'
        if 'linewidth' not in kwds:
            kwds['linewidth'] = 1
    if 'linestyle' not in mean_line_kwds:
        kwds['linestyle'] = '--'
    if 'linestyle' not in limit_lines_kwds:
        kwds['linestyle'] = ':'
    # Scatter the means virsus the differences between samples on the plot
    ax.scatter(means, diffs, **scatter_kwds)
    # draw mean line.
    ax.axhline(mean_diff, **mean_line_kwds)  

    # Annotate mean line with mean difference.
    ax.annotate('mean diff:\n{}'.format(np.round(mean_diff, 2)),
                xy=(0.99, 0.5),
                horizontalalignment='right',
                verticalalignment='center',
                fontsize=14,
                xycoords='axes fraction')

    if sd_limit > 0:
        half_ylim = (1.5 * sd_limit) * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

        limit_of_agreement = sd_limit * std_diff
        lower = mean_diff - limit_of_agreement
        upper = mean_diff + limit_of_agreement
        for j, lim in enumerate([lower, upper]):
            ax.axhline(lim, **limit_lines_kwds)
        ax.annotate('-SD{}: {}'.format(sd_limit, np.round(lower, 2)),
                    xy=(0.99, 0.07),
                    horizontalalignment='right',
                    verticalalignment='bottom',
                    fontsize=14,
                    xycoords='axes fraction')
        ax.annotate('+SD{}: {}'.format(sd_limit, np.round(upper, 2)),
                    xy=(0.99, 0.92),
                    horizontalalignment='right',
                    fontsize=14,
                    xycoords='axes fraction')

    elif sd_limit == 0:
        half_ylim = 3 * std_diff
        ax.set_ylim(mean_diff - half_ylim,
                    mean_diff + half_ylim)

    ax.set_ylabel('Difference', fontsize=15)
    ax.set_xlabel('Means', fontsize=15)
    ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.show()
    # return ax
#--------------------------------------------------------------------------------------------
# Example:
#-----------------------
# x = np.random.rand(100)
# y = np.random.rand(100)

# bland_altman_plot(x, y)