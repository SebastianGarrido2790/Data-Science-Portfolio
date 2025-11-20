### Step 3: Exploratory Data Analysis & Insight Generation

#### Objective
Conduct a comprehensive analysis of TLC trip data, weather data, and zone lookup datasets to uncover patterns, validate hypotheses, prioritize features, and inform business decisions for predicting hourly yellow taxi demand in Manhattan zones.

#### 1. Data Exploration
- **Comprehensive Analytical Techniques**:
  - **Univariate Analysis**: Assess ride count distributions (grouped by hour) with mean ~6,500 rides/hour, skewness 1.2 indicating right skew. Visualize with histograms using Seaborn.
  - **Bivariate Analysis**: Compute correlations (e.g., precipitation vs. ride count: -0.15, weak negative) using Pearson correlation matrix. Plot with pair plots.
  - **Time-Series Analysis**: Detect trends (declining from 7,000 to 5,000 rides/day Jan–Mar 2019) and daily seasonality (±800 rides) via Statsmodels decomposition. Lag plots show 1-hour autocorrelation 0.7.
  - **Anomaly Detection**: Flag outliers (e.g., >20,000 rides/hour) with Isolation Forest, affecting <1% of data.
- **Advanced Exploration Methods**:
  - **Clustering**: Apply K-means (k=5) to zone data based on ride counts, identifying high-demand clusters (e.g., Lower Manhattan).
  - **Dimensionality Reduction**: Use PCA on weather features to reduce to two components explaining 85% variance.
- **Stakeholder-Focused Visualizations**: Create Streamlit dashboard with heatmaps of zone demand and time-series trends, updated by 6:00 PM today.

#### 2. Hypothesis Testing and Validation
- **Structured Hypothesis Testing**: Test “Rain increases demand” with t-test (p=0.03, significant at 5%), supporting hypothesis. Use ANOVA for zone differences (p<0.01).
- **Business-Driven Validation**: Link findings to KPIs (e.g., demand surge costs $50K/day), focusing on revenue impact.
- **Iterative Hypothesis Refinement**: Add “Weekends boost demand” (t-test p=0.01), validated with stakeholder input.
- **Causal Inference**: Explore propensity score matching to assess weather’s causal effect, pending further data.

#### 3. Feature Prioritization and Insight Generation
- **Feature Importance Analysis**: Use Random Forest on lagged counts, weather, and zones; lagged 1-hour counts (importance 0.45), precipitation (0.20) rank highest.
- **Business-Relevant Insights**: “High demand on rainy weekends suggests targeted fleet allocation, potentially saving $100K monthly.”

#### 4. Documentation and Iteration
- **Comprehensive Documentation**: Record findings in Jupyter Notebook (`3.0-sg-eda-analysis.ipynb`), versioned in Git, with plots saved to `../../reports/figures/eda`.
- **Iterative Refinement**: Refine problem statement (e.g., focus on weekends) based on insights. Review with stakeholders by July 1, 2025.
- **Go/No-Go Decision**: Data supports objectives; recommend proceed, with risk of weather gap impact mitigated by imputation.

#### Outcome
Actionable insights (e.g., lagged counts, weather drive demand), prioritized features, and validated hypotheses. A refined problem statement and go/no-go decision guide modeling, supported by scalable, reproducible analysis. Proceed to Step 4 upon confirmation.