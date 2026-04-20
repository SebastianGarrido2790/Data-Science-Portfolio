### User Story & Problem Framing

Before writing a single line of code, we must strictly define *who* we are building this for and *how* we are translating a vague business pain into a concrete Machine Learning task.

#### 1. User Stories

We have two distinct stakeholders: the **Business User** (Fleet Manager) who consumes the insight, and the **Technical User** (You/MLOps Engineer) who maintains the system.

**Story A: The Business Value (Fleet Manager)**

> "As a **NYC Taxi Fleet Manager**, I want to **understand exactly which trip characteristics (time, location, distance) correlate with higher tips**, so that I can **optimize driver schedules and routes to maximize fleet revenue, utilizing an interactive dashboard to visualize the data and predictions.**"

**Story B: The Engineering Necessity (MLOps Engineer)**

> "As an **MLOps Engineer**, I want to **decouple the heavy data processing (5M+ rows) from the model training workflow**, so that I can **iterate on model experiments rapidly without waiting for data ingestion every time I retrain.**"

---

#### 2. Problem Framing

**The Business Context**
New York City taxi drivers complete hundreds of thousands of trips daily. While the fare is fixed by the meter, the **tip** is highly variable and constitutes a significant portion of a driver's take-home pay. Currently, drivers rely on intuition ("anecdotal evidence") to guess where the money is. This approach is inefficient and unscalable.

**The Machine Learning Formulation**
We are reframing this business problem as a **Supervised Regression Task**.

* **The Target ():** `tip_amount` (Continuous numerical value).
* **The Features ():**
    * *Temporal:* Pickup hour, day of week, month.
    * *Spatial:* Pickup/Dropoff Location IDs, Trip Distance.
    * *Transactional:* Fare amount, tolls, rate code.
* **The Constraint:** The model must handle "zero-tip" transactions (cash rides or non-tippers) without biasing the predictions for legitimate tippers.

**The Technical Challenge (Why MLOps?)**
The dataset contains **5 million rows**.

* *The "Notebook" Approach:* Loading 5M rows into Pandas inside a Jupyter Notebook for every experiment will crash the kernel and waste hours of compute time.
* *The "Pipeline" Solution:* We must process the data *once* (Feature Pipeline), save it in a compressed, machine-ready format (Parquet), and then load only the necessary bits for training (Training Pipeline).
* *The "Visualization" Need:* The solution must culminate in a Streamlit dashboard, allowing non-technical stakeholders to securely and interactively explore the model data and predictions without touching the underlying code.
