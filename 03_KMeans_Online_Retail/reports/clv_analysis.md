### 🔍 Analysis of Customer Lifetime Value (CLV) Outputs

#### 1. **Alive Probability Heatmap**

The heatmap shows:

* **X-axis (Frequency)**: Historical number of purchases.
* **Y-axis (Recency)**: Time since the last purchase (higher is older).
* **Color (Alive Probability)**: Likelihood a customer is still "alive" (i.e., active).

**Interpretation**:

* Top-left (low frequency, recent purchases): High chance of being alive.
* Bottom-right (high frequency, long time since last purchase): Low chance.
* Customers with long recency, regardless of frequency, are likely inactive.

---

#### 2. **CLV Table Insights**

* Most customers have **very low expected purchases** in the next 6 months (`expected_purchases_6_months ≈ 0`), leading to low CLVs.
* Exception: a few customers with `frequency ≥ 2`, low `recency`, and high `monetary_value` show higher CLV.

**Examples**:

* `CustomerID 12749`: Only 2 purchases, but recent (recency=122), and high value → CLV ≈ 45.98 → **champion**.
* `CustomerID 12748`: High frequency (89), but very long recency (370) → expected purchases = 0 → **low** segment.

---

#### 3. **Top Customers by Expected Purchases**

Customers with:

* `frequency = 2`
* `recency ≈ 360–368`
* `T ≈ 730`

→ all have **very low expected purchases (≈ 0.2)**. Suggests high **recency penalty** in BG/NBD.

---

#### 4. **BG/NBD Model Fit**

`<lifetimes.BetaGeoFitter: a=1.93, alpha=103.05, b=6.47, r=2.47>`

* Parameters show:

  * **Low repeat rate** overall (low `r` and `a`).
  * **High dropout likelihood** as time increases.

---

#### 5. **Gamma-Gamma Model Fit**

`<lifetimes.GammaGammaFitter: p=3.82, q=0.35, v=3.75>`

* Indicates **high variance in monetary value**.
* Correlation: `frequency ↔ monetary_value = 0.10` → weak → model use is valid.

---

#### 6. **Customer Segments (based on CLV Quantiles)**

| Segment   | Count | Mean CLV | Total CLV |
| --------- | ----- | -------- | --------- |
| low       | 458   | 0.79     | 361.16    |
| medium    | 457   | 5.14     | 2,349.76  |
| high      | 457   | 13.93    | 6,364.48  |
| champions | 457   | 43.50    | 19,878.56 |

**Observation**:
Only \~25% (champions) generate **65%+ of total CLV**, reinforcing the **Pareto principle (80/20 rule)**.

---

### ✅ Recommendations

1. **Target “champions” and “high” segments** with loyalty campaigns.
2. **Do not waste budget on “low” customers**—most are inactive.
3. **Explore recency-based reactivation strategies** (email, promotions).
4. **Consider time decay in engagement models**—BG/NBD gives sharp drop after long recency.

