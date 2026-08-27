import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from finops import pricing
from missions import m2_inference_levers, m3_purchasing


def test_cache_economics_break_even_and_validation():
    assert abs(pricing.cache_break_even_reads(1.25, 0.10) - (1.25 / 0.90)) < 1e-9
    assert pricing.cache_is_worth_it(2.0, 1.25, 0.10) is True
    assert pricing.cache_is_worth_it(1.0, 1.25, 0.10) is False


def test_cache_policy_is_applied_to_dataset():
    result = m2_inference_levers.run(verbose=False)
    assert all(x["observed_avg_reads"] > x["break_even_reads"] for x in result["cache_economics"].values())
    assert all(x["worth_it"] for x in result["cache_economics"].values())


def test_reasoning_budget_is_measured():
    reasoning = m2_inference_levers.run(verbose=False)["reasoning"]
    assert reasoning["traffic_pct"] > reasoning["cap_pct"]
    assert reasoning["cost_usd"] > 0
    assert reasoning["energy_pct"] > reasoning["traffic_pct"]
    assert reasoning["cap_daily_cost_savings"] > 0
    assert reasoning["cap_daily_wh_savings"] > 0


def test_carbon_aware_region_comparison():
    carbon = m3_purchasing.run(verbose=False)["carbon_aware"]
    assert len(carbon["region_comparison"]) == 5
    assert carbon["cleanest_region"] == "europe-north1"
    assert carbon["cheapest_region"] == "us-east-wa"
    assert carbon["carbon_saved_g"] > 0
