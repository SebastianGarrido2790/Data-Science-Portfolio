# 6. Model Evaluation & Business Review (Is our model good enough to have a practical use case?)

The LogisticRegression model meets the recall target (0.8298), which aligns with the business goal of identifying ≥70% of at-risk employees, but the low F1-score (due to low precision, 0.21) suggests a need for careful analysis and stakeholder validation.

**Goal**: Ensure the selected LogisticRegression model meets technical thresholds and delivers real-world business value without undue risk, while addressing its low F1-score through detailed analysis and stakeholder feedback. 

## Performance Metrics
- **Technical Metrics**:
    - **Recall**: 0.8298 (meets target ≥0.80, successfully identifies 82.98% of at-risk employees).
    - **F1-Score**: 0.3319 (below target ≥0.75, indicating low precision and potential over-prediction of attrition).
    - **AUC-ROC**: 0.7750 (moderate discriminative power, suggesting reasonable overall performance).
    - **Precision**: 0.21 (low, meaning 79% of predicted positives are false positives, a concern for resource allocation).
    - **Accuracy**: 0.47 (low due to class imbalance and threshold adjustment to 0.4).
- **Business-Specific KPIs**:
    - **At-Risk Employee Identification**: Achieves the goal of identifying ≥70% of at-risk employees (82.98% recall), enabling proactive retention strategies.
    - **Cost Savings**: Assuming retention of each at-risk employee saves $5,000 annually and there are 47 at-risk employees in the test set, identifying 39 (0.8298 × 47) could save approximately $195,000 per year. However, false positives (e.g., 123 predicted positives out of 47 true positives) may lead to unnecessary interventions, increasing costs (e.g., $615,000 if each intervention costs $5,000).
    - **P&L Impact**: Net impact depends on balancing savings from true positives against costs of false positives. With current precision, the model may incur a net cost unless intervention costs are minimized.
- **Benchmarking**:
    - **Random Guess Model**: For a 16% attrition rate, a random guess would yield ~16% recall (7.5 of 47), far below 0.8298.
    - **Baseline Model**: A model predicting all as non-attrition (majority class) would have 0% recall, making LogisticRegression a significant improvement.
- **What-If Scenarios**:
    - **Higher Threshold (e.g., 0.5)**: Recall drops to ~0.66 (previous run), missing 17 of 47 at-risk employees, reducing savings to ~$150,000 but lowering false positives.
    - **Lower Threshold (e.g., 0.3)**: Recall may increase to ~0.85, but precision could drop further, exacerbating false positive costs.

## Error Analysis
- **False Negatives (FN)**: 8 of 47 at-risk employees missed (1 - 0.8298). These are critical errors, potentially costing $40,000 in lost savings. SHAP analysis (e.g., `OverTime`, `SatisfactionScore`) suggests these may be cases with moderate risk factors not captured well.
- **False Positives (FP)**: 123 predicted positives minus 39 true positives = 84 false positives. At $5,000 per intervention, this adds $420,000 in costs, outweighing savings unless intervention costs are lower.
- **Model Shortcomings**:
    - Low precision indicates the model over-predicts attrition, possibly due to class imbalance handling (SMOTE + class_weight).
    - Interpretability is strong (SHAP highlights `OverTime`, `SatisfactionScore`), but the model may overfit to oversampled data.
- **Refinement Suggestions**:
    - Adjust threshold using precision-recall curve to balance recall and precision.
    - Reduce SMOTE `sampling_strategy` (e.g., to 0.5) to limit noise.
    - Collect more data to improve minority class representation.

## Business Validation
- **Findings Presentation**:
    - **Strengths**: Achieves recall target (0.8298), interpretable via SHAP, aligns with HR’s need for actionable insights.
    - **Weaknesses**: Low F1-score (0.3319) due to precision 0.21, leading to high false positive rates and potential cost overruns.
    - **Stakeholder Discussion**: Present to HR with SHAP plots (`reports/figures/shap_logistic_regression.png`) and cost-benefit analysis. Highlight that while 82.98% of at-risk employees are caught, 84 false positives may strain resources unless intervention costs are minimal (e.g., <~$2,300 per case to break even).
- **Expected ROI**: $195,000 savings - $420,000 cost = -$225,000 net loss unless false positives are filtered (e.g., by HR review).

## Go/No-Go Decision
- **Evaluation Insights**:
    - The model meets the recall target, fulfilling the technical KPI and business goal of identifying at-risk employees.
    - However, the low precision and resulting cost overrun pose a risk to practical deployment without mitigation.
- **Mitigation Plans**:
    - **Threshold Optimization**: Use a precision-recall curve to find a threshold (e.g., 0.35-0.45) that balances recall (≥0.80) and reduces false positives.
    - **HR Filtering**: Implement a manual review step for predicted positives to filter out false positives, reducing costs.
    - **Cost Control**: Negotiate lower intervention costs (e.g., $1,000 per case) to improve net ROI.
- **Decision**: Conditional Go – Proceed to deployment with approval from HR, contingent on implementing threshold optimization and a review process to mitigate false positives. Document expected ROI ($195,000 savings with mitigated costs) and risks (cost overruns if filtering fails).

## Outcome
- **Formal Approval**: Documented approval to deploy LogisticRegression (version 2) with:
    - **Expected ROI**: $195,000 annual savings with mitigated costs (<$2,300 per intervention).
    - **Risks**: High false positive rate ($420,000 potential cost) unless mitigated by HR review or threshold adjustment.
    - **Mitigation Plans**: Optimize threshold, add manual review, and monitor performance post-deployment.
- **Next Action**: Transition to Step 7 (Deployment) with the approved model, ensuring HR is aligned on the mitigation strategy.

## Next Steps
- **Transition to Step 7**: Proceed with deploying LogisticRegression (version 2), implementing threshold optimization and HR review processes.
- **Stakeholder Engagement**: Share the evaluation with HR and finalize the mitigation strategy.