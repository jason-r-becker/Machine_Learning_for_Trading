"""MC2-P1: Market simulator - grading script.

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

import numpy as np
import pandas as pd
from collections import namedtuple

from util import get_data
from util import get_orders_data_file

# Student code
main_code = "marketsim"  # module name to import

# Test cases
MarketsimTestCase = namedtuple('MarketsimTestCase', ['description', 'group', 'inputs', 'outputs'])
marketsim_test_cases = [
    MarketsimTestCase(
        description="Orders 1",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-01-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 245 ,
            last_day_portval = 1168048.2605,
            sharpe_ratio = 1.1246688622 ,
            avg_daily_ret = 0.000695379308027
        )
    ),
    MarketsimTestCase(
        description="Orders 2",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-02-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 245 ,
            last_day_portval = 941728.68375 ,
            sharpe_ratio = -0.484584226147 ,
            avg_daily_ret = -0.000202950697606
        )
    ),
    MarketsimTestCase(
        description="Orders 3",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-03-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 240 ,
            last_day_portval = 852671.278 ,
            sharpe_ratio = -1.0315603247 ,
            avg_daily_ret = -0.000611776749722
        )
    ),
    MarketsimTestCase(
        description="Orders 4",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-04-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 146 ,
            last_day_portval = 757106.846 ,
            sharpe_ratio = -1.95300766073,
            avg_daily_ret =  -0.00178910543474
        )
    ),
    MarketsimTestCase(
        description="Orders 5",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-05-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 296 ,
            last_day_portval = 1285570.62 ,
            sharpe_ratio = 1.93114474121 ,
            avg_daily_ret = 0.000882663269112
        )
    ),
    MarketsimTestCase(
        description="Orders 6",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-06-noleverage.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 210 ,
            last_day_portval = 909893.372,
            sharpe_ratio = -1.35910878739,
            avg_daily_ret =  -0.000433113220844
        )
    ),
    MarketsimTestCase(
        description="Orders 7 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-07-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 237 ,
            last_day_portval = 1100291.9735 ,
            sharpe_ratio = 1.98512994113 ,
            avg_daily_ret = 0.000410688184214
        )
    ),
    MarketsimTestCase(
        description="Orders 8 (modified)",
        group='basic',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-08-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            num_days = 229 ,
            last_day_portval = 1062228.657 ,
            sharpe_ratio =  0.791123631185,
            avg_daily_ret = 0.000282839500507
        )
    ),
    #######################
    # Leverage test cases #
    #######################
    MarketsimTestCase(
        description="Orders 11 - Leveraged SELL (modified)",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-11-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1035922.85
        )
    ),
    MarketsimTestCase(
        description="Orders 12 - Leveraged BUY (modified)",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-12-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1024418.1015
        )
    ),
    MarketsimTestCase(
        description="Wiki leverage example #1",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-1.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1027011.95
        )
    ),
    MarketsimTestCase(
        description="Modified wiki leverage example #2",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-2-modified.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1045542.55
        )
    ),
    MarketsimTestCase(
        description="Wiki leverage example #3",
        group='leverage',
        inputs=dict(
            orders_file=os.path.join('orders', 'orders-leverage-3.csv'),
            start_val=1000000
        ),
        outputs=dict(
            last_day_portval = 1026835.13825
        )
    ),
    MarketsimTestCase(
        description="author() test",
        group='author',
        inputs=None,
        outputs=None
    ),
]

seconds_per_test_case = 10  # execution time limit

# Grading parameters (picked up by module-level grading fixtures)
max_points = 100.0 
html_pre_block = True  # surround comments with HTML <pre> tag (for T-Square comments field)

# Test functon(s)
@pytest.mark.parametrize("description,group,inputs,outputs", marketsim_test_cases)
def test_marketsim(description, group, inputs, outputs, grader):
    """Test compute_portvals() returns correct daily portfolio values.

    Requires test description, test case group, inputs, expected outputs, and a grader fixture.
    """

    points_earned = 0.0  # initialize points for this test case
    try:
        # Try to import student code (only once)
        if not main_code in globals():
            import importlib
            # * Import module
            mod = importlib.import_module(main_code)
            globals()[main_code] = mod
            # * Import methods to test
            for m in ['compute_portvals']:
                globals()[m] = getattr(mod, m)

        incorrect = False
        msgs = []

        if group == 'author':
            try:
                # globals()['author'] = getattr(marketsim, 'author')
                auth_string = run_with_timeout(marketsim.author,seconds_per_test_case,(),{})
                if auth_string == 'tb34':
                    incorrect = True
                    msgs.append("   Incorrect author name (tb34)")
                    points_earned = -20
                elif auth_string == '':
                    incorrect = True
                    msgs.append("   Empty author name")
                    points_earned = -20
            except Exception as e:
                incorrect = True
                msgs.append("   Exception occured when calling author() method: {}".format(e))
                points_earned = -20
        else:
            # Unpack test case
            orders_file = inputs['orders_file']
            start_val = inputs['start_val']

            portvals = None
            fullpath_orders_file = get_orders_data_file(orders_file)
            portvals = run_with_timeout(compute_portvals,seconds_per_test_case,(fullpath_orders_file,start_val),{})
            # Verify against expected outputs and assign points

            # * Check return type is correct, coax into Series
            assert (type(portvals) == pd.Series) or (type(portvals) == pd.DataFrame and len(portvals.columns) == 1), "You must return a Series or single-column DataFrame!"
            if type(portvals) == pd.DataFrame:
                portvals = portvals[portvals.columns[0]]  # convert single-column DataFrame to Series
            if group == 'basic':
                if len(portvals) != outputs['num_days']:
                    incorrect=True
                    msgs.append("   Incorrect number of days: {}, expected {}".format(len(portvals), outputs['num_days']))
                else:
                    points_earned += 2.0
                if abs(portvals[-1]-outputs['last_day_portval']) > (0.001*outputs['last_day_portval']):
                    incorrect=True
                    msgs.append("   Incorrect final value: {}, expected {}".format(portvals[-1],outputs['last_day_portval']))
                else:
                    points_earned += 5.0
                adr,sr = get_stats(portvals)
                if abs(sr-outputs['sharpe_ratio']) > abs(0.001*outputs['sharpe_ratio']):
                    incorrect=True
                    msgs.append("   Incorrect sharpe ratio: {}, expected {}".format(sr,outputs['sharpe_ratio']))
                else:
                    points_earned += 1.5
                if abs(adr-outputs['avg_daily_ret']) > abs(0.001*outputs['avg_daily_ret']):
                    incorrect=True
                    msgs.append("   Incorrect avg daily return: {}, expected {}".format(adr,outputs['avg_daily_ret']))
                else:
                    points_earned += 1.0
            elif group=='leverage':
                if abs(portvals[-1]-outputs['last_day_portval']) > (0.001*outputs['last_day_portval']):
                    incorrect = True
                    msgs.append("   Incorrect final value: {}, expected {}".format(portvals[-1],outputs['last_day_portval']))
                else:
                    points_earned += 1.0
        if incorrect:
            # inputs_str = "    orders_file: {}\n" \
                         # "    start_val: {}\n".format(orders_file, start_val)
            raise IncorrectOutput, "Test failed on one or more output criteria.\n  Inputs:\n{}\n  Failures:\n{}".format(inputs, "\n".join(msgs))
    except Exception as e:
        # Test result: failed
        msg = "Test case description: {}\n".format(description)
        
        # Generate a filtered stacktrace, only showing erroneous lines in student file(s)
        
        tb_list = tb.extract_tb(sys.exc_info()[2])
        if 'grading_traceback' in dir(e):
            tb_list = e.grading_traceback
        for i in xrange(len(tb_list)):
            row = tb_list[i]
            tb_list[i] = (os.path.basename(row[0]), row[1], row[2], row[3])  # show only filename instead of long absolute path
        tb_list = [row for row in tb_list if row[0] == 'marketsim.py']
        if tb_list:
            msg += "Traceback:\n"
            msg += ''.join(tb.format_list(tb_list))  # contains newlines
        msg += "{}: {}".format(e.__class__.__name__, e.message)

        # Report failure result to grader, with stacktrace
        grader.add_result(GradeResult(outcome='failed', points=points_earned, msg=msg))
        raise
    else:
        # Test result: passed (no exceptions)
        grader.add_result(GradeResult(outcome='passed', points=points_earned, msg=None))

def get_stats(port_val):
    daily_rets = (port_val / port_val.shift(1)) - 1
    daily_rets = daily_rets[1:]
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sharpe_ratio = np.sqrt(252) * daily_rets.mean() / std_daily_ret
    return avg_daily_ret, sharpe_ratio

if __name__ == "__main__":
    pytest.main(["-s", __file__])
