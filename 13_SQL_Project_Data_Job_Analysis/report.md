# 🔍 The Analysis

Each query for this project aimed at investigating specific aspects of the data scientist job market. Each subsection includes strategic insights to guide job seekers, educators, or hiring teams.

## 1.  💰 Top-Paying Data Science Jobs

The dataset contains 1,086 job postings for data scientist roles, with the following relevant columns:

* `job_id`: Unique identifier for each job post
* `job_title`: Title of the data science role
* `salary_year_avg`: Average yearly salary for the role
* `company_name`: Hiring company
* `skills`: A single skill associated with the role (note: only one skill per row)

The `skills` column contains one skill per row, so the same job may appear in multiple rows with different skills. I will now analyze the `skills` column to provide insights such as:

* Most frequently requested skills
* Average salary per skill
* Skill co-occurrence (if meaningful despite one-skill-per-row)

### 💡 Insights

#### 🔹 Most Frequently Requested Skills

1. **Python** – 173 postings
2. **SQL** – 122 postings
3. **R** – 90 postings
4. **Tableau** – 49 postings
5. **SAS** – 48 postings

#### 🔹 Top Skills by Frequency and Their Average Salaries

| Skill      | Frequency | Average Salary (USD) |
| ---------- | --------- | -------------------- |
| Python     | 173       | 148,262              |
| SQL        | 122       | 147,203              |
| R          | 90        | 138,340              |
| Tableau    | 49        | 137,484              |
| SAS        | 48        | 126,063              |
| Spark      | 36        | 155,516              |
| Pytorch    | 26        | 144,891              |
| TensorFlow | 26        | 143,937              |
| AWS        | 24        | 134,378              |
| Excel      | 22        | 139,205              |

#### 🔹 Noteworthy High-Paying Skills (Despite Lower Frequency)

* **Go** – \$169,126 (15 listings)
* **Pandas** – \$166,283 (16 listings)
* **Scikit-learn** – \$164,191 (16 listings)
* **Snowflake** – \$158,692 (15 listings)
* **Azure** – \$155,826 (17 listings)

### Interpretation

* **Core data science tools** (Python, SQL, R) dominate in demand.
* **Machine learning libraries** (e.g., TensorFlow, PyTorch, Scikit-learn) also rank well in both demand and salary.
* **Cloud platforms** (AWS, Azure, Databricks, Snowflake) are fewer in frequency but tend to offer high-paying roles.
* **Emerging or niche tools** like Go and Pandas, while less frequently mentioned, are associated with premium salaries.

The chart above visualizes the **top 20 most requested data science skills**:

* **Blue bars** represent how frequently each skill appeared in job postings.
* **Red line** shows the **average salary** associated with each skill.

### Key Takeaways:

* **Python** and **SQL** dominate in demand.
* **Pandas**, **Scikit-learn**, **Go**, and **Azure** stand out for **high salaries** despite being less frequently listed.
* Skills like **Spark**, **Databricks**, and **Snowflake** strike a balance between demand and compensation.

---

## 2. 💰 Top 30 Highest-Paying Data Science Roles

The dataset includes the **top 30 highest-paying data science roles** available remotely in the **United States**, all with non-null salary values. Key columns for analysis:

* `job_title`: Titles of the roles
* `salary_year_avg`: Average yearly salary
* `company_name`: Employer

Let’s proceed to analyze:

1. Salary distribution across job titles
2. Highest-paying titles and companies
3. Group similar job titles for pattern recognition

![Top 30 Paying Remote Roles in the U.S.](figure\Top_30_Paying_Remote_Data_Science_Roles.png)

The visualization displays the **top 30 highest-paying remote data science roles** in the U.S. (2023), sorted by salary and distinguished by company.

### 💡 Insights

#### 🔹 Salary Range

* Salaries range from approximately **\$170,000** to **\$262,500**.
* The **highest-paid role** is a **Senior Data Scientist - TS Clearance** at **Harnham** (\$262,500).
* Other high-paying roles are offered by **BCG X**, **Intelletec**, **Glocomms**, and **Airtable**, with salaries above \$230,000.

#### 🔹 Recurring Seniority Levels

* Titles including **“Principal”** or **“Senior”** dominate the top-paying list.
* Examples:

  * *Principal Data Scientist*
  * *Senior Data Scientist*
  * *Lead Data Scientist*
* This indicates that seniority significantly influences compensation.

#### 🔹 Company Highlights

* **BCG X**, **Meta**, and **Harnham** appear multiple times, suggesting consistent high compensation strategies.
* Several startups and specialized consultancies (e.g., **Glocomms**, **Intelletec**) also pay competitive salaries.

---

## 1. 🏢 **Company Salaries (Top-Paying Employers)**

| Company          | Avg Salary (USD) | Job Count |
| ---------------- | ---------------- | --------- |
| Selby Jennings   | \$419,167        | 3         |
| Reddit           | \$254,575        | 5         |
| Braintrust       | \$225,000        | 3         |
| Lawrence Harvey  | \$223,333        | 6         |
| Home Depot / THD | \$212,000        | 5         |
| Meta             | \$192,750        | 6         |

### 💡 Insights:

* **Selby Jennings** stands out with an exceptionally high average salary, suggesting specialized or senior-level roles.
* Big tech (e.g. **Reddit**, **Meta**, **Atlassian**, **MongoDB**) offers competitive packages, aligning with industry expectations.
* **Recruitment/staffing firms** (Selby Jennings, Lawrence Harvey, Storm3, Averity) appear frequently, indicating many high-paying roles are through agencies.

**Implication**: Candidates aiming for high salaries should not overlook roles from recruitment firms and should also target companies with repeated, high-value postings.

---

## 2. ⏰ **Job Schedule Type vs. Salary**

| Schedule Type                        | Avg Salary (USD) | Count |
| ------------------------------------ | ---------------- | ----- |
| Full-time                            | \$146,019        | 902   |
| Full-time + Part-time                | \$144,850        | 6     |
| Part-time                            | \$135,000        | 2     |
| Contractor                           | \$112,742        | 33    |
| Internship                           | \$101,840        | 1     |
| Full-time, Part-time, and Internship | \$43,030         | 1     |

### 💡 Insights:

* **Full-time roles** dominate both in volume and salary, reaffirming their status as the most stable and lucrative path.
* **Contractor positions** offer significantly lower pay on average, possibly due to fewer benefits and less job security.
* **Hybrid schedules** (Full-time + other types) don’t consistently offer higher pay, suggesting salary isn't directly tied to flexibility.

**Implication**: For maximizing income, pursuing full-time roles remains the most effective strategy. Contractor roles may suit those seeking flexibility, but likely at a financial trade-off.

---

## 3. 🌍 **Best U.S. Locations for Remote Data Science Jobs**

| Location   | Avg Salary (USD) | Job Count |
| ---------- | ---------------- | --------- |
| California | \$150,687        | 192       |
| Texas      | \$150,234        | 157       |
| New York   | \$143,412        | 124       |
| Illinois   | \$143,172        | 181       |
| Florida    | \$140,179        | 30        |

### 💡 Insights:

* **California and Texas** lead in both salary and volume, suggesting they are the top remote hubs for data science.
* **Illinois** surprisingly surpasses New York in job count, which may reflect the rise of tech-related roles in non-coastal states.
* High salaries across all five states reinforce the viability of remote work while still reflecting geographic influence on job trends.

**Implication**: Job seekers can broaden their focus beyond traditional tech hubs (California, NY) to include states like Texas and Illinois, which are becoming remote job powerhouses.

---

## 4. 🎓 **Impact of Degree Requirement on Salary**

| No Degree Mention | Avg Salary (USD) | Job Count |
| ----------------- | ---------------- | --------- |
| True              | \$148,562        | 21        |
| False             | \$144,304        | 932       |

### 💡 Insights:

* Surprisingly, **jobs that do *not* require a degree** pay **\~\$4,000 more** on average.
* However, the **sample size is very small (21 jobs)**, which limits statistical confidence.

**Implication**: While promising for non-traditional candidates, the data suggests employers are cautiously opening up to skills-first hiring. More data over time could validate this trend.

---

## 5. 🏥 **Health Insurance Mention and Salary**

| Health Insurance | Avg Salary (USD) | Job Count |
| ---------------- | ---------------- | --------- |
| Yes              | \$145,905        | 552       |
| No               | \$142,324        | 401       |

### 💡 Insights:

* Jobs **with health insurance** pay slightly more, though the difference is modest (\~\$3,600).
* This could reflect **larger or more established companies** offering both higher pay and benefits.

**Implication**: Benefits still serve as a partial proxy for job quality. Candidates should consider total compensation (including perks), not just base salary.

---

## 6. 🧠 **Most Common Skill Combinations**

![Top Skill Pair Co-occurrences in Job Postings](figure\skills_combination.png)

### 💡 Insights:

* **Python + SQL** is the most prevalent and foundational skill combo in data science.
* **Python + R** and **SQL + R** indicate a strong preference for statistical programming alongside database querying.
* **Python + Cloud (AWS/Azure)** and **Python + Spark/TensorFlow** reveal demand for scalable and ML-driven workflows.
* **Visualization tools (Tableau)** frequently appear alongside core programming skills.

**Implication**:

* **Core stack for data scientists**: Python + SQL + R
* **For machine learning engineers**: Add TensorFlow, Spark
* **For analytics-focused roles**: Add Tableau, SAS

These combinations reflect both the technical depth and breadth expected of modern data scientists.

---

### 📌 Summary of Strategic Takeaways

| Area               | Recommendation                                                                    |
| ------------------ | --------------------------------------------------------------------------------- |
| Top Companies      | Target firms like Selby Jennings, Reddit, Meta, Atlassian for higher salaries.    |
| Schedule Types     | Full-time jobs remain the most lucrative and prevalent.                           |
| Location Strategy  | Don’t limit to the coasts—Texas and Illinois also offer excellent prospects.      |
| Degree Flexibility | Some high-paying roles are accessible without formal degrees—skills matter.       |
| Benefits           | Health insurance often coincides with better-paying, more secure roles.           |
| Skill Development  | Prioritize Python, SQL, R, and pair them with cloud, ML, and visualization tools. |



