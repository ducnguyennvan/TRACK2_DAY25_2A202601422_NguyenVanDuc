"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    interruptible_energy_wh = 0.0
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        if interruptible:
            interruptible_energy_wh += gpu_hours * num(c["watts"])
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    region_comparison = []
    for region, intensity in sustainability.REGION_CARBON.items():
        region_comparison.append({
            "region": region,
            "price_per_kwh": sustainability.REGION_PRICE_KWH[region],
            "carbon_g_per_kwh": intensity,
            "electricity_cost_usd": round(sustainability.energy_cost_usd(interruptible_energy_wh, region), 2),
            "carbon_g": round(sustainability.carbon_g(interruptible_energy_wh, region), 1),
        })
    cleanest = min(region_comparison, key=lambda x: x["carbon_g_per_kwh"])
    cheapest = min(region_comparison, key=lambda x: x["price_per_kwh"])
    baseline_region = next(x for x in region_comparison if x["region"] == "us-east-1")
    carbon_aware = {
        "energy_kwh": round(interruptible_energy_wh / 1000.0, 1),
        "baseline_region": "us-east-1",
        "cleanest_region": cleanest["region"],
        "cheapest_region": cheapest["region"],
        "carbon_saved_g": round(baseline_region["carbon_g"] - cleanest["carbon_g"], 1),
        "carbon_reduction_pct": round((1 - cleanest["carbon_g"] / baseline_region["carbon_g"]) * 100, 1),
        "region_comparison": region_comparison,
    }

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print("\ncarbon-aware scheduling (interruptible jobs):")
        print(f"{'region':20}{'$/kWh':>8}{'gCO2/kWh':>11}{'electricity':>13}{'carbon kg':>12}")
        for r in region_comparison:
            print(f"{r['region']:20}{r['price_per_kwh']:>8.3f}{r['carbon_g_per_kwh']:>11.0f}"
                  f"${r['electricity_cost_usd']:>11,.2f}{r['carbon_g']/1000:>12,.1f}")
        print(f"move us-east-1 -> {cleanest['region']}: save {carbon_aware['carbon_saved_g']/1000:,.1f} kgCO2e "
              f"({carbon_aware['carbon_reduction_pct']:.1f}%); cheapest electricity is {cheapest['region']}")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "carbon_aware": carbon_aware}


if __name__ == "__main__":
    run()
