"""
Run the full backtest simulation and print results.
    python -m scripts.backtest
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, data_gen
from src.backtest import compute_false_positive_costs, plot_backtest, run_backtest, score_test_set
from src.model import time_respecting_split
from src.pipeline import load_artifacts


def main():
    model, calibrator, feature_cols, history_df = load_artifacts()
    raw = data_gen.generate_disputes()
    _, _, test_raw = time_respecting_split(raw)

    print(f"Scoring {len(test_raw)} test disputes through the full pipeline...")
    scored = score_test_set(test_raw, model, calibrator, feature_cols, history_df)

    print("\n=== FALSE-POSITIVE COST ANALYSIS ===")
    fp = compute_false_positive_costs(scored)
    print(json.dumps(fp, indent=2))

    print("\n=== BACKTEST: 3-STRATEGY COMPARISON ===")
    bt = run_backtest(scored)
    print(json.dumps(bt, indent=2))

    print(f"\n{'=' * 60}")
    print("KEY NUMBERS FOR THE VIDEO")
    print(f"{'=' * 60}")
    print(f"Accept all losses:    ₹{bt['strategy_accept_all_inr']:>12,.2f}")
    print(f"Contest all outcome:  ₹{bt['strategy_contest_all_inr']:>12,.2f}")
    print(f"This system outcome:  ₹{bt['strategy_system_inr']:>12,.2f}")
    print(f"Savings vs accept:    ₹{bt['savings_vs_accept_all_inr']:>12,.2f} ({bt['savings_pct_vs_accept_all']}%)")
    print(f"Savings vs contest:   ₹{bt['savings_vs_contest_all_inr']:>12,.2f}")
    print(f"False positive rate:  {fp['false_positive_rate']:.1%}")
    print(f"False positive cost:  ₹{fp['false_positive_cost_inr']:>12,.2f}")
    print(
        f"Labor savings:        ₹{bt['labor_savings_inr']:>12,.2f} "
        f"({bt['n_disputes'] - bt['analyst_contested_from_escalated']} cases automated)"
    )
    print(f"{'=' * 60}")

    plot_backtest(bt)
    print(f"\nBacktest chart saved to {config.ARTIFACTS_DIR / 'backtest_comparison.png'}")

    with open(config.ARTIFACTS_DIR / "backtest_results.json", "w") as f:
        json.dump({"false_positive_analysis": fp, "backtest": bt}, f, indent=2)


if __name__ == "__main__":
    main()
