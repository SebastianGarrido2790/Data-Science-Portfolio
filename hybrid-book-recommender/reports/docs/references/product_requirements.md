## **Product Requirements Document (PRD)**

In a professional setting, writing code without a clear definition of *value* and *constraints* is a recipe for wasted engineering effort. We need to frame our project not just as a technical exercise, but as a solution to a specific business problem with defined constraints.

### **1. The User Story & Problem Framing**
**The Context:**
Imagine you are the Lead Data Scientist at **"Libri,"** a fictional boutique online bookstore struggling with user retention.

**The Problem:**
New users sign up but leave within 60 seconds because the homepage shows generic bestsellers they have already read (The "Harry Potter" effect). They experience **choice paralysis** and do not feel "understood" by the platform. The current Collaborative Filtering system fails for these users because they have zero interaction history (The "Cold Start" problem).

**The User Story:**

> *"As a new user (Alex) who just finished a niche sci-fi novel, I want to paste a description of what I liked and immediately see relevant suggestions, so that I can buy my next book without browsing through hundreds of generic lists."*

---

### **2. Success Metrics (KPIs)**
We distinguish between what the business cares about (Money/Engagement) and what the model optimizes (Math).

#### **A. Business Metric (The North Star)**
Since this is a portfolio project, we cannot measure real-time sales. However, we will design the system to optimize for:

* **Click-Through Rate (CTR):** The percentage of recommended books that a user clicks on to view details.
* **Session Duration:** We aim to increase the time a user spends exploring the "Recommended for You" section.

#### **B. Machine Learning Metrics (The Proxies)**
These are the offline metrics we calculate during the **Phase 3: Modeling** stage to predict business success.

* **For Collaborative Filtering (Track A):**
* **Precision@K (e.g., @10):** Of the top 10 books recommended, how many are actually relevant (based on hold-out test data)?
* **Personalization Score:** Measures how different one user's recommendations are from another's (preventing the "everyone sees the same bestsellers" issue).
* **For Content-Based/LLM (Track B):**
* **Cosine Similarity:** Measures how semantically close the recommended book's embedding is to the user's query/history.
* **Coverage:** The percentage of items in the catalog that the system is capable of recommending. (The LLM system should have 100% coverage, unlike CF which usually misses the "long tail").

---

### **3. Scalability & Constraints (The "Real-World" Reality Check)**

#### **I. Costs (Budgeting)**
We must treat compute and API calls as money.

* **LLM Costs (Gemini API):**
* **Constraint:** 7,000 books \times ~200 tokens/description = ~1.4 Million tokens.
* **Estimation:** Gemini 1.5 Flash is highly affordable (often free tier eligible or cents per million tokens). We must cache these embeddings in ChromaDB so we only pay **once** per book, not every time a user searches.
* **Infrastructure (AWS):**
* **Constraint:** We will use the **AWS Free Tier** (t2.micro or t3.micro).
* **Implication:** Our Docker container must be memory-efficient. We cannot load a massive 10GB model into RAM. This justifies using an API-based LLM (Gemini) over a locally hosted Llama 3 model.

#### **II. Fairness, Bias & Privacy*** **Popularity Bias:** 
The Collaborative Filtering model (KNN) inherently favors popular items. If 80% of users rated "The Great Gatsby," the model will spam it to everyone.
* **Mitigation:** The **Hybrid** nature of our architecture acts as a bias countermeasure. The Content-Based (LLM) track recommends books based on *content*, not popularity, surfacing hidden gems.
* **Representation Bias:** Do the book descriptions in the dataset reflect diverse authors?
* **Check:** During **Phase 1 (EDA)**, we will check the distribution of authors. If the LLM consistently tags "Romance" books with "feminine" sentiment or "Sci-Fi" with "masculine" sentiment, we need to note this ethical risk in our report.

#### **III. Data Availability*** **The Constraint:** 
We are using the static `7k-books-with-metadata` dataset. In a real startup, data arrives continuously (streams).
* **The Workaround:** To simulate a real-world environment, we will design our **Ingestion Pipeline (Stage 00)** to be idempotent. It should be able to run daily, checking for *new* rows in the source CSV and only processing the delta (the difference), rather than retraining from scratch every time.

### **Summary of Success Definition**
| Aspect | Portfolio Definition |
| --- | --- |
| **Problem** | "Cold Start" for new users & lack of semantic search. |
| **Business Goal** | Simulate increased CTR and discovery of niche titles. |
| **ML Metric** | Precision@10 (Relevance) and Catalog Coverage (Diversity). |
| **Cost Cap** | AWS Free Tier + Gemini Free Tier (Caching required). |
| **Fairness** | Use Hybrid Hybrid approach to mitigate popularity bias. |

### **Next Step**
With the problem framed and metrics defined, we can now confidently move to the "Lab" phase.
