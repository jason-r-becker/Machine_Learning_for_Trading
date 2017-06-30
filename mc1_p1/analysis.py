"""MC1-P1: Analyze a portfolio."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from util import get_data, plot_data
from matplotlib import style

style.use('ggplot')

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality
def assess_portfolio(sd=dt.datetime(2008, 1, 1), ed=dt.datetime(2009, 1, 1),
                     syms=['GOOG', 'AAPL', 'GLD', 'XOM'],
                     allocs=[0.1, 0.2, 0.3, 0.4],
                     sv=1000000, rfr=0.0, sf=252.0,
                     gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later

    # Get daily portfolio value
    # port_val = prices_SPY  # add code here to compute daily portfolio values
    normed = prices / prices.ix[0, :]
    alloced = normed * allocs
    pos_vals = alloced * sv
    port_vals = pos_vals.sum(axis=1)

    # Get daily returns
    daily_ret = port_vals.copy()
    daily_ret[1:] = (daily_ret[1:] / daily_ret[:-1].values) - 1
    daily_ret.ix[0] = 0

    # Get portfolio statistics (note: std_daily_ret = volatility)
    # cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1]  # add code here to compute stats
    cr = (port_vals[-1] / port_vals[0]) - 1
    adr = daily_ret[1:].mean()
    sddr = daily_ret[1:].std()
    sr = np.sqrt(sf) * (adr - rfr) / sddr

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        normed_SPY = prices_SPY / prices_SPY.ix[0, :]
        df_temp = pd.concat([alloced.sum(axis=1), normed_SPY], keys=['Portfolio', 'SPY'], axis=1)
        df_temp.plot()
        plt.title('Daily Portfolio Value Compared to S&P 500', fontsize=20)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Price', fontsize=18)
        plt.draw()
        pass

    # Add code here to properly compute end value
    ev = (1 + cr) * sv

    return cr, adr, sddr, sr, ev


def try_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!

    start_val = 1000000

    # Test Cases

    # Start Date: 2010-01-01
    # End Date: 2010-12-31
    # Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    # Allocations: [0.2, 0.3, 0.4, 0.1]
    # Sharpe Ratio: 1.51819243641
    # Volatility (stdev of daily returns): 0.0100104028
    # Average Daily Return: 0.000957366234238
    # Cumulative Return: 0.255646784534
    # start_date = dt.datetime(2010, 1, 1)
    # end_date = dt.datetime(2010, 12, 31)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    # allocations = [0.2, 0.3, 0.4, 0.1]

    # Start Date: 2010-01-01
    # End Date: 2010-12-31
    # Symbols: ['AXP', 'HPQ', 'IBM', 'HNZ']
    # Allocations: [0.0, 0.0, 0.0, 1.0]
    # Sharpe Ratio: 1.30798398744
    # Volatility (stdev of daily returns): 0.00926153128768
    # Average Daily Return: 0.000763106152672
    # Cumulative Return: 0.198105963655
    # start_date = dt.datetime(2010,1,1)
    # end_date = dt.datetime(2010,12,31)
    # symbols = ['AXP', 'HPQ', 'IBM', 'HNZ']
    # allocations = [0.0, 0.0, 0.0, 1.0]

    # Start Date: 2010-06-01
    # End Date: 2010-12-31
    # Symbols: ['GOOG', 'AAPL', 'GLD', 'XOM']
    # Allocations: [0.2, 0.3, 0.4, 0.1]
    # Sharpe Ratio: 2.21259766672
    # Volatility (stdev of daily returns): 0.00929734619707
    # Average Daily Return: 0.00129586924366
    # # Cumulative Return: 0.205113938792
    # start_date = dt.datetime(2010,6,1)
    # end_date = dt.datetime(2010,12,31)
    # symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    # allocations = [0.2, 0.3, 0.4, 0.1]

    # start_date: 2010-06-01 00:00:00
    # end_date: 2011-06-01 00:00:00
    # symbols: ['AAPL', 'GLD', 'GOOG', 'XOM']
    # allocs: [0.1, 0.4, 0.5, 0.0]
    # start_val: 1000000
    # sharpe_ratio: 1.10895144722 (expected: -6.87191373641)
    # start_date = dt.datetime(2010,6,1)
    # end_date = dt.datetime(2011,6,1)
    # symbols = ['AAPL', 'GLD', 'GOOG', 'XOM']
    # allocations = [0.1, 0.4, 0.5, 0.0]

    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd=start_date, ed=end_date,
                                             syms=symbols,
                                             allocs=allocations,
                                             sv=start_val,
                                             gen_plot=True)

    # Print statistics
    print "Start Date:", start_date
    print "End Date:", end_date
    print "Symbols:", symbols
    print "Allocations:", allocations
    print "Sharpe Ratio:", sr
    print "Volatility (stdev of daily returns):", sddr
    print "Average Daily Return:", adr
    print "Cumulative Return:", cr


if __name__ == "__main__":
    try_code()
    plt.show()
