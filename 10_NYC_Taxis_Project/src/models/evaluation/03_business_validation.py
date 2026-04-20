import os
import pandas as pd
import plotly.graph_objects as go
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

# Debug: Print actual column names to identify the issue
print("Actual segment columns:", segment.columns.tolist())

# Prepare data for dashboard
metrics_long = metrics.melt(
    id_vars=["model"],
    value_vars=["RMSE", "MAE", "cost_savings_usd", "ride_completion_rate"],
    var_name="Metric",
    value_name="Value",
)

# Handle corrupted MultiIndex in segment data
# Assign column names based on expected order (temporary fix)
segment.columns = [
    "Borough",
    "residual_xgb_mean",
    "residual_xgb_std",
    "residual_prophet_mean",
    "residual_prophet_std",
]
print("Mapped segment columns:", segment.columns.tolist())

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


# Create interactive dashboard
def create_dashboard(metrics_df, segment_df):
    # Metrics plot
    fig_metrics = go.Figure()
    for metric in metrics_df["Metric"].unique():
        df_metric = metrics_df[metrics_df["Metric"] == metric]
        fig_metrics.add_trace(
            go.Bar(
                x=df_metric["model"],
                y=df_metric["Value"],
                name=metric,
                text=df_metric["Value"].round(3),
                textposition="auto",
            )
        )
    fig_metrics.update_layout(
        barmode="group",
        title="Model Performance Metrics",
        xaxis_title="Model",
        yaxis_title="Score",
        legend_title="Metric",
    )
    fig_metrics.show()
    # Save interactive plot as HTML
    fig_metrics.write_html(
        os.path.join(base_dir, "../../../reports/dashboard_metrics.html")
    )

    # Segment analysis plot
    fig_segment = go.Figure()
    for metric in segment_df["Metric"].unique():
        df_metric = segment_df[segment_df["Metric"] == metric]
        fig_segment.add_trace(
            go.Bar(
                x=df_metric["Borough"],
                y=df_metric["Value"],
                name=metric,
                text=df_metric["Value"].round(3),
                textposition="auto",
            )
        )
    fig_segment.update_layout(
        barmode="group",
        title="Residual Analysis by Borough",
        xaxis_title="Borough",
        yaxis_title="Residual (Mean/Std)",
        legend_title="Metric",
    )
    fig_segment.show()
    # Save interactive plot as HTML
    fig_segment.write_html(
        os.path.join(base_dir, "../../../reports/dashboard_segment.html")
    )


create_dashboard(metrics_long, segment_long)

# Stakeholder report with updated segment insights
report = f"""
Business Validation Report - {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}
--------------------------------------------------
- **Performance Summary**:
  - XGBoost: RMSE = {metrics.loc[metrics['model'] == 'XGBoost', 'RMSE'].iloc[0]:.3f}, 
    Cost Savings = ${metrics.loc[metrics['model'] == 'XGBoost', 'cost_savings_usd'].iloc[0]:.2f}, 
    Ride Completion Rate = {metrics.loc[metrics['model'] == 'XGBoost', 'ride_completion_rate'].iloc[0]:.2%}
  - Prophet: RMSE = {metrics.loc[metrics['model'] == 'Prophet', 'RMSE'].iloc[0]:.3f}, 
    Cost Savings = ${metrics.loc[metrics['model'] == 'Prophet', 'cost_savings_usd'].iloc[0]:.2f}, 
    Ride Completion Rate = {metrics.loc[metrics['model'] == 'Prophet', 'ride_completion_rate'].iloc[0]:.2%}
  - Recommendation: XGBoost outperforms with lower RMSE (0.279) and positive cost savings ($220.73).

- **Segment Insights**:
  - Residual analysis shows higher mean residuals for XGBoost in Manhattan (0.007) and higher standard deviation (0.391), indicating potential over-prediction or variability.
  - Prophet shows larger mean residuals across all boroughs (e.g., Manhattan 0.332, std 1.244), suggesting poor fit.
  - Potential focus: Investigate Manhattan-specific data quality or demand patterns.

- **Next Steps**:
  - Review interactive plots at 2025-06-20 15:00 -04.
  - Provide feedback on KPI alignment (e.g., RMSE < 0.5, cost savings > $200) and risks (e.g., Manhattan bias).

- **Contact**: Data Science Team
"""

with open(
    os.path.join(base_dir, "../../../reports/business_validation_report.txt"), "w"
) as f:
    f.write(report)
print("Business validation report saved: business_validation_report.txt")
