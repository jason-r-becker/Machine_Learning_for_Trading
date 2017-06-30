"""MC2-P1: Market simulator."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util import get_data


def author():
    return 'Jason R Becker'


def compute_portvals(orders_file="./orders/orders.csv", start_val=1000000):
    # this is the function the autograder will call to test your code
    # TODO: Your code here

    # Read orders file
    orders_df = pd.read_csv(orders_file, index_col='Date', parse_dates=True, na_values=['nan'])
    orders_df.sort_index()

    # Apply leverage test if required
    if ('leverage' in orders_file):
        cmsn = 9.95
        mrt_impct = 0.005
        lev = True

    end = False
    while not end:
        orders_df.sort_index(inplace=True)
        start_date = orders_df.index[0]
        end_date = orders_df.index[-1]

        # Collect price data for each ticker in order
        df_prices = get_data(orders_df.Symbol.unique().tolist(), pd.date_range(start_date, end_date))
        df_prices = df_prices.drop('SPY', 1)  # remove SPY
        df_prices['cash'] = 1

        # Track trade data
        df_trades = df_prices.copy()
        df_trades[:] = 0

        # Populate trade dataframe
        for i, date in enumerate(orders_df.index):
            # Get order information
            sym = orders_df.Symbol[i]
            if orders_df.Order[i] == 'BUY':
                order = 1
            else:
                order = -1
            shares = orders_df.Shares[i]

            # Calculate change in shares and cash
            df_trades[sym][date] += order * shares
            df_trades['cash'][date] += -order * shares * df_prices[sym][date]

        # Track total holdings
        df_holdings = df_prices.copy()
        df_holdings[:] = 0

        # Include starting value
        df_holdings['cash'][0] = start_val

        # Update first day of holdings
        for c in df_trades.columns:
            df_holdings[c][0] += df_trades[c][0]

        # Update every day, adding new day's trade information with previous day's holdings
        for i in range(1, len(df_trades.index)):
            for c in df_trades.columns:
                df_holdings[c][i] += df_trades[c][i] + df_holdings[c][i-1]

        # Track monetary values
        df_values = df_prices.mul(df_holdings)

        # Define port_val
        port_val = df_values.sum(axis=1)

        # Calculate leverage:
        df_stocks = df_values.copy()
        df_stocks.drop('cash', 1, inplace=True)
        stock_val = df_stocks.sum(axis=1)
        leverage = stock_val.div(port_val)
        leverage = pd.DataFrame(leverage)
        leverage.columns = ['lev']

        # Update order sheet if necessary
        new_orders = pd.concat([orders_df, leverage], axis=1, join='inner')
        update_orders = new_orders[new_orders['lev'] < 2]

        if len(new_orders) == len(update_orders):
            end = True
        else:
            orders_df = update_orders.copy()
            orders_df = orders_df.drop('lev', axis=1)

    return port_val


def try_code():
    # this is a helper function you can use to test your code
    # note that during autograding his function will not be called.
    # Define input parameters

    of = "./orders/orders-leverage-2.csv"
    sv = 1000000

    # Process orders
    portvals = compute_portvals(orders_file=of, start_val=sv)
    portvals = pd.DataFrame(portvals)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # just get the first column
    else:
        "warning, code did not return a DataFrame"

    # Get SPY performance over same period
    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    spy = get_data([], pd.date_range(start_date, end_date))

    # Get SPY stats
    daily_ret_SPY = spy.copy()
    daily_ret_SPY[1:] = (daily_ret_SPY[1:] / daily_ret_SPY[:-1].values) - 1
    daily_ret_SPY.ix[0] = 0

    cr_SPY = (spy.ix[-1] / spy.ix[0]) - 1
    adr_SPY = daily_ret_SPY[1:].mean()
    stdr_SPY = daily_ret_SPY[1:].std()
    sr_SPY = np.sqrt(252) * adr_SPY / stdr_SPY

    # Get fund stats
    daily_ret = portvals.copy()
    daily_ret[1:] = (daily_ret[1:] / daily_ret[:-1].values) - 1
    daily_ret.ix[0] = 0

    cr = (portvals[-1] / portvals[0]) - 1
    adr = daily_ret[1:].mean()
    stdr = daily_ret[1:].std()
    sr = np.sqrt(252) * adr / stdr

    def clean(stat):
        return round(float(stat), 8)

    # Compare portfolio against $SPX
    print "Date Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(clean(sr))
    print "Sharpe Ratio of SPY : {}".format(clean(sr_SPY))
    print
    print "Cumulative Return of Fund: {}".format(clean(cr))
    print "Cumulative Return of SPY : {}".format(clean(cr_SPY))
    print
    print "Standard Deviation of Fund: {}".format(clean(stdr))
    print "Standard Deviation of SPY : {}".format(clean(stdr_SPY))
    print
    print "Average Daily Return of Fund: {}".format(clean(adr))
    print "Average Daily Return of SPY : {}".format(clean(adr_SPY))
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot comparison to SPY
    normed_SPY = spy / spy.ix[0, :]
    normed_portvals = portvals / portvals[0]
    plt.figure()
    plt.plot(spy.index, normed_portvals)
    plt.plot(spy.index, normed_SPY.ix[:, 0])
    plt.legend(['Fund', 'S&P 500'])
    plt.title('Daily Portfolio Value Compared to S&P 500', fontsize=20)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Price', fontsize=18)
    plt.draw()
    pass

if __name__ == "__main__":
    try_code()
    plt.show()