"""MC3-P3: Q-learning & Dyna - grading script.

Usage:
- Switch to a student feedback directory first (will write "points.txt" and "comments.txt" in pwd).
- Run this script with both ml4t/ and student solution in PYTHONPATH, e.g.:
    PYTHONPATH=ml4t:MC1-P2/jdoe7 python ml4t/mc2_p1_grading/grade_marketsim.py
"""

import pytest
from grading.grading import grader, GradeResult, run_with_timeout, IncorrectOutput

import os
import sys
import traceback as tb

import datetime as dt
import numpy as np
import pandas as pd
from collections import namedtuple

import time
import util
import random

# Student modules to import
main_code = ['StrategyLearner',]  # module name to import

# Test cases
StrategyTestCase = namedtuple('Strategy', ['description','insample_args','outsample_args','benchmark_type','benchmark','train_time','test_time','max_time','seed'])
strategy_test_cases = [
    StrategyTestCase(
        description="ML4T-220",
        insample_args=dict(symbol="ML4T-220",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000),
        outsample_args=dict(symbol="ML4T-220",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000),
        benchmark_type='clean',
        benchmark=1.0, #benchmark updated Apr 24 2017
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="AAPL",
        insample_args=dict(symbol="AAPL",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000),
        outsample_args=dict(symbol="AAPL",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000),
        benchmark_type='stock',
        benchmark=0.00312594756827, #benchmark computed Jul 16 2017
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="SINE_FAST_NOISE",
        insample_args=dict(symbol="SINE_FAST_NOISE",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000),
        outsample_args=dict(symbol="SINE_FAST_NOISE",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000),
        benchmark_type='noisy',
        benchmark=2.0, #benchmark updated Apr 24 2017
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
    StrategyTestCase(
        description="UNH - In sample",
        insample_args=dict(symbol="UNH",sd=dt.datetime(2008,1,1),ed=dt.datetime(2009,12,31),sv=100000),
        outsample_args=dict(symbol="UNH",sd=dt.datetime(2010,1,1),ed=dt.datetime(2011,12,31),sv=100000),
        benchmark_type='stock',
        benchmark= -0.00498631271416, #benchmark computed Jul 16 2017
        train_time=25,
        test_time=5,
        max_time=60,
        seed=1481090000
        ),
]

max_points = 60.0 
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

MAX_HOLDINGS = 200

# Test functon(s)
@pytest.mark.parametrize("description, insample_args, outsample_args, benchmark_type, benchmark, train_time, test_time, max_time, seed", strategy_test_cases)
def test_strategy(description, insample_args, outsample_args, benchmark_type, benchmark, train_time, test_time, max_time, seed, grader):
    """Test StrategyLearner.

    Requires test description, insample args (dict), outsample args (dict), benchmark_type (str), benchmark (float)
    max time (seconds), points for this test case (int), random seed (long), and a grader fixture.
    """
    points_earned = 0.0  # initialize points for this test case
    try:
        incorrect = True
        if not 'StrategyLearner' in globals():
            import importlib
            m = importlib.import_module('StrategyLearner')
            globals()['StrategyLearner'] = m
        outsample_cr_to_beat = None
        if benchmark_type == 'clean':
            outsample_cr_to_beat = benchmark
        def timeoutwrapper_strategylearner():
            learner = StrategyLearner.StrategyLearner(verbose=False)
            tmp = time.time()
            learner.addEvidence(**insample_args)
            train_t = time.time()-tmp
            tmp = time.time()
            insample_trades_1 = learner.testPolicy(**insample_args)
            test_t = time.time()-tmp
            insample_trades_2 = learner.testPolicy(**insample_args)
            tmp = time.time()
            outsample_trades = learner.testPolicy(**outsample_args)
            out_test_t = time.time()-tmp
            return insample_trades_1, insample_trades_2, outsample_trades, train_t, test_t, out_test_t
        #Set fixed seed for repetability
        np.random.seed(seed)
        random.seed(seed)
        msgs = []
        in_trades_1, in_trades_2, out_trades, train_t, test_t, out_test_t = run_with_timeout(timeoutwrapper_strategylearner,max_time,(),{})
        incorrect = False
        if ((in_trades_1.abs()>2*MAX_HOLDINGS).any()[0]) or ((in_trades_2.abs()>2*MAX_HOLDINGS).any()[0]) or ((out_trades.abs()>2*MAX_HOLDINGS).any()[0]):
            incorrect = True
            msgs.append("  illegal trade. abs(trades) > {}".format(2*MAX_HOLDINGS))
        if ((in_trades_1.cumsum().abs()>MAX_HOLDINGS).any()[0]) or ((in_trades_2.cumsum().abs()>MAX_HOLDINGS).any()[0]) or ((out_trades.cumsum().abs()>MAX_HOLDINGS).any()[0]):
            incorrect = True
            msgs.append("  holdings more than {} long or short".format(MAX_HOLDINGS))
        if not(incorrect):
            if train_t>train_time:
                incorrect=True
                msgs.append("  addEvidence() took {} seconds, max allowed {}".format(train_t,train_time))
            else:
                points_earned += 1.0
            if test_t > test_time:
                incorrect = True
                msgs.append("  testPolicy() took {} seconds, max allowed {}".format(test_t,test_time))
            else:
                points_earned += 2.0
            if not((in_trades_1 == in_trades_2).all()[0]):
                incorrect = True
                mismatches = in_trades_1.join(in_trades_2,how='outer',lsuffix='1',rsuffix='2')
                mismatches = mismatches[mismatches.ix[:,0]!=mismatches.ix[:,1]]
                msgs.append("  consecutive calls to testPolicy() with same input did not produce same output:")
                msgs.append("  Mismatched trades:\n {}".format(mismatches))
            else:
                points_earned += 2.0
            student_insample_cr = evalPolicy2(insample_args['symbol'],in_trades_1,insample_args['sv'])
            student_outsample_cr = evalPolicy2(outsample_args['symbol'],out_trades, outsample_args['sv'])
            if student_insample_cr < benchmark:
                incorrect = True
                msgs.append("  in-sample return ({}) did not beat benchmark ({})".format(student_insample_cr,benchmark))
            else:
                points_earned += 5.0
            if outsample_cr_to_beat is None:
                if out_test_t > test_time:
                    incorrect = True
                    msgs.append("  out-sample took {} seconds, max of {}".format(out_test_t,test_time))
                else:
                    points_earned += 5.0
            else:
                if student_outsample_cr < outsample_cr_to_beat:
                    incorrect = True
                    msgs.append("  out-sample return ({}) did not beat benchmark ({})".format(student_outsample_cr,outsample_cr_to_beat))
                else:
                    points_earned += 5.0
        if incorrect:
            inputs_str = "    insample_args: {}\n" \
                         "    outsample_args: {}\n" \
                         "    benchmark_type: {}\n" \
                         "    benchmark: {}\n" \
                         "    train_time: {}\n" \
                         "    test_time: {}\n" \
                         "    max_time: {}\n" \
                         "    seed: {}\n".format(insample_args, outsample_args, benchmark_type, benchmark, train_time, test_time, max_time,seed)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs_str, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        tb_list = tb.extract_tb(sys.exc_info()[2])
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        # tb_list = [row for row in tb_list if row[0] in ['QLearner.py','StrategyLearner.py']]
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        elif 'grading_traceback' in dir(e):
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(e.grading_traceback))
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

def compute_benchmark(sd,ed,sv,symbol):
    date_idx = util.get_data([symbol,],pd.date_range(sd,ed)).index
    orders = pd.DataFrame(index=date_idx)
    orders['orders'] = 0; orders['orders'][0] = MAX_HOLDINGS; orders['orders'][-1] = -MAX_HOLDINGS
    return evalPolicy2(symbol,orders,sv)

def evalPolicy(student_trades,sym_prices,startval):
    ending_cash = startval - student_trades.mul(sym_prices,axis=0).sum()
    ending_stocks = student_trades.sum()*sym_prices.ix[-1]
    return float((ending_cash+ending_stocks)/startval)-1.0

def evalPolicy2(symbol, student_trades, startval):
    orders_df = pd.DataFrame(columns=['Shares','Order','Symbol'])
    for row_idx in student_trades.index:
        nshares = student_trades.loc[row_idx][0]
        if nshares == 0:
            continue
        order = 'sell' if nshares < 0 else 'buy'
        new_row = pd.DataFrame([[abs(nshares),order,symbol],],columns=['Shares','Order','Symbol'],index=[row_idx,])
        orders_df = orders_df.append(new_row)
    portvals = compute_portvals(orders_df,startval)
    return float(portvals[-1]/startval)-1

def compute_portvals(orders_df, startval):
    """Simulate the market for the given date range and orders file."""
    MARKET_IMPACT_PENALTY = 50 * 0.0001
    COMMISSION_COST = 9.95    
    symbols = []
    orders = []
    orders_df = orders_df.sort_index()
    start_date=orders_df.index[0]
    end_date = orders_df.index[-1]
    for date, order in orders_df.iterrows():
        shares = order['Shares']
        action = order['Order']
        symbol = order['Symbol']
        if action.lower() == 'sell':
            shares *= -1
        order = (date, symbol, shares)
        orders.append(order)
        symbols.append(symbol)
    symbols = list(set(symbols))
    dates = pd.date_range(start_date, end_date)
    prices_all = util.get_data(symbols, dates)
    prices = prices_all[symbols]
    prices = prices.fillna(method='ffill').fillna(method='bfill')
    prices['_CASH'] = 1.0
    trades = pd.DataFrame(index=prices.index, columns=symbols)
    trades = trades.fillna(0)
    cash = pd.Series(index=prices.index)
    cash = cash.fillna(0)
    cash.ix[0] = startval
    for date, symbol, shares in orders:
        price = prices[symbol][date]
        val = shares * price
        # transaction cost model
        val += COMMISSION_COST + (pd.np.abs(shares)*price*MARKET_IMPACT_PENALTY)
        positions = prices.ix[date] * trades.sum()
        totalcash = cash.sum()
        if (date < prices.index.min()) or (date > prices.index.max()):
            continue
        trades[symbol][date] += shares
        cash[date] -= val
    trades['_CASH'] = cash
    holdings = trades.cumsum()
    df_portvals = (prices * holdings).sum(axis=1)
    return df_portvals

if __name__ == "__main__":
    pytest.main(["-s", __file__])
