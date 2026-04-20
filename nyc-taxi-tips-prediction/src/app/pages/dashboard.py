"""
Dashboard page for the NYC Taxi Tips Streamlit app.

This module provides the visualizations and metrics for model performance
and latest batch inferences.
"""

from typing import Any

import pandas as pd
import plotly.express as px
import requests
import streamlit as st


def render_dashboard(
    metrics: dict[str, Any],
    predictions_df: pd.DataFrame,
    params: dict[str, Any] | None,
    api_url: str,
) -> None:
    """Renders the Dashboard & Evaluation page.

    Args:
        metrics: Dictionary containing model evaluation metrics.
        predictions_df: DataFrame containing batch inference results.
        params: Optional dictionary of configuration parameters.
        api_url: Base URL of the FastAPI backend for fetching feature importance.
    """
    st.header("Model Performance & Batch Inferences")

    # Metrics Row
    col1, col2, col3 = st.columns(3)
    val_mae = metrics.get("test_mae", 0.0)
    val_mse = metrics.get("test_mse", 0.0)
    val_r2 = metrics.get("test_r2", 0.0)

    col1.metric(
        "Test MAE (Mean Absolute Error)",
        f"${val_mae:.2f}",
        delta="- Error indicates high precision",
        delta_color="normal",
    )
    col2.metric(
        "Test R² (Variance Explained)",
        f"{val_r2:.4f}",
        delta="> 0.90 is excellent",
        delta_color="normal",
    )
    col3.metric("Test MSE (Mean Squared Error)", f"{val_mse:.4f}")

    if params and "Training" in params and "selection_metrics" in params.get("Training", {}):
        selection_metrics = params["Training"]["selection_metrics"]
        weights_str = "   |   ".join(
            [f"**{k.upper()}**: {v}" for k, v in selection_metrics.items()]
        )
        st.info(
            f"**🏆 Champion Selection Weights:**   {weights_str}  \n"
            "*(Models are scored and ranked automatically based on these parameters)*",
            icon="⚖️",
        )
        st.caption(
            "💡 *Note: A weight of 1.0 is considered the 'best' performance for that metric.*"
        )

    st.markdown("---")

    # Feature Importance and Predictions Plot Row
    col_feat, col_pred = st.columns([1, 1])

    with col_feat:
        st.subheader("Feature Importance")
        st.markdown("What drives tipping behavior?")

        # Fetch Feature Importance from FastAPI Backend
        feature_names, importances = None, None
        try:
            res = requests.get(f"{api_url}/v1/feature-importance", timeout=2)
            if res.status_code == 200:
                data = res.json()
                feature_names = data.get("features")
                importances = data.get("importances")
        except requests.exceptions.RequestException:
            pass

        if importances is not None and feature_names is not None:
            feat_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})

            # Normalize the importance to sum to 100 (Relative Percentage)
            total_importance = feat_df["Importance"].sum()
            if total_importance > 0:
                feat_df["Importance"] = ((feat_df["Importance"] / total_importance) * 100).round(4)

            num_features = st.slider(
                "Number of Top Features to Display",
                min_value=3,
                max_value=20,
                value=10,
                step=1,
            )

            feat_df = feat_df.sort_values(by="Importance", ascending=True).tail(
                num_features
            )  # Top N

            fig = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title=f"Top {num_features} Drivers of Taxi Tips",
                color="Importance",
                color_continuous_scale="Viridis",
                template="plotly_dark",
                labels={"Importance": "Relative Importance (%)"},
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "💡 *Importance is normalized as a percentage of predictive power across features.*"
            )
        else:
            st.info("Feature importance not supported for the current champion model type.")

    with col_pred:
        st.subheader("Latest Batch Predictions")

        sample_size_limit = min(10000, len(predictions_df)) if len(predictions_df) > 0 else 100
        num_batch_sample = st.slider(
            "Distribution Sample Size",
            min_value=100,
            max_value=sample_size_limit,
            value=min(5000, len(predictions_df)),
            step=100,
        )

        st.markdown(
            f"Displaying a sample of {num_batch_sample:,} out of "
            f"{len(predictions_df):,} recent inferences."
        )

        # Quick distribution plot of predictions
        if "predicted_tip" in predictions_df.columns:
            fig2 = px.histogram(
                predictions_df.sample(num_batch_sample)
                if num_batch_sample <= len(predictions_df)
                else predictions_df,
                x="predicted_tip",
                nbins=50,
                title="Distribution of Predicted Tips (Sample)",
                color_discrete_sequence=["#FFD700"],
                template="plotly_dark",
            )
            st.plotly_chart(fig2, width="stretch")

            # Show actual data with enhanced visual context
            st.markdown("##### 🧾 Inferences Ledger")

            ledger_sample_limit = min(1000, len(predictions_df)) if len(predictions_df) > 0 else 10
            num_ledger_sample = st.slider(
                "Ledger Random Sample",
                min_value=10,
                max_value=ledger_sample_limit,
                value=min(100, len(predictions_df)),
                step=10,
            )

            disp_df = (
                predictions_df.sample(num_ledger_sample).copy()
                if num_ledger_sample <= len(predictions_df)
                else predictions_df.copy()
            )
            if "VendorID" in disp_df.columns:
                # Map real NYC taxi service providers for better business context
                vendor_map = {1: "🚗 Creative Mobile", 2: "🚕 VeriFone Inc"}
                disp_df["Vendor"] = disp_df["VendorID"].map(lambda x: vendor_map.get(x, str(x)))

                # Reorder columns to put Vendor first
                cols = ["Vendor", "predicted_tip"] + [
                    c for c in disp_df.columns if c not in ["Vendor", "predicted_tip", "VendorID"]
                ]
                disp_df = disp_df[cols]

            st.dataframe(
                disp_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "predicted_tip": st.column_config.NumberColumn(
                        "Predicted Tip",
                        help="Model's estimated tip amount in USD",
                        format="$ %.2f",
                    ),
                    "Vendor": st.column_config.TextColumn(
                        "Taxi Provider", help="Technology Service Provider"
                    ),
                },
            )
            st.caption(
                "💡 *Vendor IDs from the TLC data are mapped to their service providers:*\n"
                "* **1:** Creative Mobile Technologies, LLC\n"
                "* **2:** VeriFone Inc."
            )
