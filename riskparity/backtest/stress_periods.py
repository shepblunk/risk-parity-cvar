"""
Known market-stress windows used to slice backtest results.

Keeping these as explicit, dated constants (rather than an ad hoc
"look at the chart") makes the stress analysis reproducible and
citable in the README -- e.g. "vol-ERC and CVaR-ERC risk contributions
diverge most during {name}, see figure X".

TODO: sanity-check exact start/end dates against your actual price
history (some of these are approximate and should be tightened, e.g.
to the ETF universe's own drawdown dates rather than generic dates).
"""

STRESS_PERIODS = {
    "gfc_2008": ("2008-09-01", "2009-03-31"),
    "euro_crisis_2011": ("2011-07-01", "2011-10-31"),
    "taper_tantrum_2013": ("2013-05-01", "2013-09-30"),
    "vol_shock_2018": ("2018-10-01", "2018-12-31"),
    "covid_crash_2020": ("2020-02-19", "2020-03-23"),
    "rate_shock_2022": ("2022-01-01", "2022-10-31"),
}
