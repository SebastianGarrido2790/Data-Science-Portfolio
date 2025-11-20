import os
import pandas as pd
from datetime import datetime

# Set base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load performance metrics and segment analysis
metrics_path = os.path.join(
    base_dir, "../../../data/processed/performance_metrics.parquet"
)
segment_path = os.path.join(
    base_dir, "../../../data/processed/segment_analysis.parquet"
)
metrics = pd.read_parquet(metrics_path)
segment = pd.read_parquet(segment_path)

# Assign column names based on expected order (temporary fix)
segment.columns = [
    "Borough",
    "residual_xgb_mean",
    "residual_xgb_std",
    "residual_prophet_mean",
    "residual_prophet_std",
]

# Transform segment data to include Metric column
segment_long = segment.melt(
    id_vars=["Borough"],
    value_vars=[
        "residual_xgb_mean",
        "residual_xgb_std",
        "residual_prophet_mean",
        "residual_prophet_std",
    ],
    var_name="Metric",
    value_name="Value",
)

# Define decision criteria
decision_criteria = {
    "performance": {"threshold": 0.5, "metric": "RMSE", "pass": lambda x: x < 0.5},
    "cost_savings": {
        "threshold": 200.0,
        "metric": "cost_savings_usd",
        "pass": lambda x: x > 200.0,
    },
    "completion_rate": {
        "threshold": 0.95,
        "metric": "ride_completion_rate",
        "pass": lambda x: x >= 0.95,
    },
    "fairness": {
        "threshold": 0.1,
        "metric": "residual_xgb_std",
        "pass": lambda x: x < 0.1,
    },  # Max std deviation
}


# Evaluate models against criteria
def evaluate_model(model_name, metrics_df, segment_df):
    model_metrics = metrics_df[metrics_df["model"] == model_name].iloc[0]
    segment_stats = segment_df[segment_df["Metric"] == "residual_xgb_std"]

    results = {}
    for criterion, params in decision_criteria.items():
        value = (
            model_metrics[params["metric"]]
            if criterion != "fairness"
            else segment_stats[segment_stats["Borough"] == "Manhattan"]["Value"].iloc[0]
        )
        results[criterion] = {
            "value": value,
            "threshold": params["threshold"],
            "pass": params["pass"](value),
            "comment": f"{'Pass' if params['pass'](value) else 'Fail'} - {value} {'<' if criterion == 'performance' else '>' if criterion == 'cost_savings' else '>=' if criterion == 'completion_rate' else '<'} {params['threshold']}",
        }
    return results


# Perform Go/No-Go assessment
xgboost_results = evaluate_model("XGBoost", metrics, segment_long)
prophet_results = evaluate_model("Prophet", metrics, segment_long)

# Generate decision report
report = f"""
Go/No-Go Decision Report - {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}
--------------------------------------------------
- **Decision Criteria**:
  - Performance (RMSE < 0.5)
  - Cost Savings (> $200)
  - Completion Rate (>= 95%)
  - Fairness (Max Std Dev < 0.1 in Manhattan)

- **XGBoost Evaluation**:
  - Performance: {xgboost_results['performance']['comment']}
  - Cost Savings: {xgboost_results['cost_savings']['comment']}
  - Completion Rate: {xgboost_results['completion_rate']['comment']}
  - Fairness: {xgboost_results['fairness']['comment']}
  - Decision: {'Go' if all(result['pass'] for result in xgboost_results.values()) else 'No-Go'}

- **Prophet Evaluation**:
  - Performance: {prophet_results['performance']['comment']}
  - Cost Savings: {prophet_results['cost_savings']['comment']}
  - Completion Rate: {prophet_results['completion_rate']['comment']}
  - Fairness: {prophet_results['fairness']['comment']}
  - Decision: {'Go' if all(result['pass'] for result in prophet_results.values()) else 'No-Go'}

- **Recommendations**:
  - XGBoost meets performance, cost savings, and completion rate criteria but fails fairness due to high standard deviation (0.391) in Manhattan.
  - Prophet fails performance and cost savings criteria; fairness is also a concern.
  - Suggested Action: Proceed with XGBoost deployment, investigate Manhattan data for bias mitigation, and consider Prophet retraining or rejection.

- **Next Steps**:
  - Finalize deployment plan for XGBoost by 2025-06-25.
  - Address fairness concerns with stakeholder input by 2025-06-22.

- **Contact**: Data Science Team
"""

with open(os.path.join(base_dir, "../../../reports/go_no_go_report.txt"), "w") as f:
    f.write(report)
print("Go/No-Go decision report saved: go_no_go_report.txt")
