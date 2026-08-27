"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    lie_rows = ", ".join(
        f"`{x['gpu_id']}` ({x['gpu_util_pct']:.1f}% GPU-Util, MFU {x['mfu']:.3f}, MBU {x['mbu']:.3f})"
        for x in r1["lies"]
    )
    cache_lines = "\n".join(
        f"| {tier} | {e['write_cost_multiplier']:.2f}x | {e['break_even_reads']:.2f} | "
        f"{e['observed_avg_reads']:.2f} | {'Yes' if e['worth_it'] else 'No'} |"
        for tier, e in r2["cache_economics"].items()
    )
    reason = r2["reasoning"]
    ca = r3["carbon_aware"]
    region_lines = "\n".join(
        f"| {x['region']} | ${x['price_per_kwh']:.3f} | {x['carbon_g_per_kwh']:.0f} | "
        f"${x['electricity_cost_usd']:,.2f} | {x['carbon_g']/1000:,.1f} |"
        for x in ca["region_comparison"]
    )
    top_team, top_cost = max(r4["by_team"].items(), key=lambda x: x[1])
    analysis_sections = [
        """## Why GPU-Util can lie

The flagged devices are: %s. `nvidia-smi` GPU-Util measures the fraction of time a kernel is active, not useful FLOPs delivered. A kernel can remain active while stalled on HBM, launch overhead, I/O, or small batches. NimbusAI therefore pays the full GPU-hour even when MFU shows that only a small fraction of rented compute is producing model work. Validate memory-bound decode with MBU and optimize batching/model placement before assuming a busy clock is efficient.""" % lie_rows,
        """## Prioritized action plan

1. **Move checkpointable jobs to spot and reserve only steady services** because purchasing is the largest modeled dollar lever (**$%s/month**). Monitor interruption/rework cost before expanding commitments.
2. **Apply inference routing, cache, and batch controls** because they require no long commitment and immediately lower the product KPI from **$%.3f to $%.3f/1M-token**.
3. **Terminate idle instances and right-size flagged GPUs after an SLO/capacity test**. The cheaper GPU must still satisfy VRAM and bandwidth needs; hourly price alone is not enough.
4. **Govern allocation**: tag coverage is %.1f%%, so the 80%% chargeback gate is open. `%s` is currently the largest attributed team at $%.2f/day.""" % (
            f"{purchasing_savings:,.0f}", r2["baseline_per_m"], r2["optimized_per_m"],
            r4["tag_coverage"] * 100, top_team, top_cost),
        """## Extension 3 — Cache economics

The CSV has no prefix identifier, so `(project, route_tier)` is used as a conservative proxy for a reusable system/RAG prefix. A write is worthwhile only when `reads × (1 − read discount) > write cost`.

| Tier | Write cost | Break-even reads | Observed avg reads | Cache enabled |
|---|---:|---:|---:|---|
%s

Both tiers use the same 1.25x write multiplier and 0.10x cached-read price, hence the same %.2f-read threshold; dollar savings per token are larger for the large tier. The observed reuse exceeds the threshold, so cache savings are counted.""" % (
            cache_lines, next(iter(r2["cache_economics"].values()))["break_even_reads"]),
        """## Extension 4 — Reasoning budget

Reasoning is **%.1f%% of requests** but **%.1f%% of optimized inference cost** and **%.1f%% of energy** (reasoning: $%.2f and %.1f Wh/day; normal: $%.2f and %.1f Wh/day). Energy is disproportionate because the simulation applies the documented 80x reasoning multiplier in addition to longer outputs.

Routing rule: enable reasoning only for requests whose complexity classifier is above 0.70 or whose first-pass confidence is below 0.60; otherwise use the normal route. The workload already satisfies a 10%% cap, so a stricter measurable budget of **%.0f%% of traffic** is estimated to save **$%.2f/day (%s/month)** and **%.1f Wh/day**. The dollar counterfactual caps output at the median non-reasoning output for the same tier; the energy counterfactual removes the 80x multiplier for only the excess traffic.""" % (
            reason["traffic_pct"], reason["cost_pct"], reason["energy_pct"], reason["cost_usd"],
            reason["energy_wh"], reason["normal_cost_usd"], reason["normal_energy_wh"], reason["cap_pct"],
            reason["cap_daily_cost_savings"], f"${reason['cap_daily_cost_savings']*DAYS:,.2f}", reason["cap_daily_wh_savings"]),
        """## Extension 5 — Carbon-aware scheduling

Interruptible jobs consume **%.1f kWh/month** in this workload model.

| Region | $/kWh | gCO2/kWh | Electricity/month | Carbon kgCO2e |
|---|---:|---:|---:|---:|
%s

Moving these jobs from `us-east-1` to **`%s`** reduces emissions by **%.1f kgCO2e (%.1f%%)**. The cheapest electricity is **`%s`**, so “optimal” depends on the objective: `%s` minimizes carbon while `%s` minimizes energy cost. Before migration, check data-residency and latency because the cleanest region may be farther from users; training is a better relocation candidate than latency-sensitive inference.""" % (
            ca["energy_kwh"], region_lines, ca["cleanest_region"], ca["carbon_saved_g"] / 1000,
            ca["carbon_reduction_pct"], ca["cheapest_region"], ca["cleanest_region"], ca["cheapest_region"]),
        """## Measurement notes

- All spend figures are monthly projections from the deterministic June-2026 snapshot; inference logs represent one day and are multiplied by 30.
- Lever amounts are modeled opportunities, not invoices. Validate quality, latency, capacity, checkpoint frequency, and regional constraints in a staged rollout.
- FOCUS export and showback should remain in place after optimization so realized savings can be compared with this baseline.""",
    ]

    md = report.build_report(
        baseline, optimized, levers, sustainability=sust,
        unit_economics={
            "baseline_per_m": r2["baseline_per_m"],
            "optimized_per_m": r2["optimized_per_m"],
            "savings_pct": r2["savings_pct"],
        },
        analysis_sections=analysis_sections,
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
