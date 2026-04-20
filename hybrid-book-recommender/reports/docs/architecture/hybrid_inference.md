# **Hybrid Inference Engine Architecture**

## **1. Executive Summary**
The `HybridRecommender` (located in `src/models/hybrid_recommender.py`) acts as the "Brain" of the system. It solves the **Generic Recommendation Problem** (recommending popular but irrelevant items) and the **Long-Tail Problem** (recommending relevant but low-quality items) by combining two scoring signals: deep semantic intent and historical popularity data.

---

## **2. The Hybrid Algorithm**

The final score for a book is determined by a **Dynamic Context-Aware Strategy**:

### **A. Default Mode (General Search)**
If no specific emotional tone is selected, the system prioritizes a balance of relevance and quality:
$$ Score_{final} = (1 - Distance_{cosine}) + \alpha \cdot (\frac{Rating_{avg}}{5.0}) $$

### **B. Outcome-Based Mode (Tone Filtering)**
If a user selects an emotional target (e.g., "Joy"), the system pivots to prioritizing that specific emotional probability:
1.  **Filter:** Only consider books where $P(Emotion) > 0$.
2.  **Sort:** Rank primarily by the specific **Emotion Probability** ($P_{emotion}$) in descending order.
3.  **Secondary Sort:** Use Semantic Similarity as a tie-breaker.

---

## **3. Code Architecture & Principles**

### **A. Component Isolation & DI**
Following the **Antigravity Stack** standards, the `HybridRecommender` follows a **Dependency Injection (DI)** pattern. It does not instantiate its own configuration; instead, it receives a pre-validated `InferenceConfig` entity from the `ConfigurationManager`.

### **B. Initialization (`__init__`)**
*   **Decoupled Embeddings**: Uses `EmbeddingFactory` to instantiate the search model. Note: The `inference` model configuration can now be scaled independently of the `trainer` configuration in `params.yaml`.
*   **VectorDB Connection**: Establishes a read-only connection to the persisted `chroma_db` (ChromaDB Vector Store).
*   **Metadata Store**: Loads `toned_books.csv` (Enriched + Toned) into a Pandas DataFrame. This serves as an **O(1) lookup table** for ratings AND emotional probabilities, ensuring sub-millisecond scoring latency.

### **C. Recommendation Flow (`recommend`)**
1.  **Semantic Retrieval**:
    *   The user query is converted into a vector.
    *   ChromaDB returns the top $K \times 3$ candidates to ensure a healthy pool.
2.  **Entity Linking**:
    *   Extracts the ISBN and hydrates the candidate with metadata (Ratings, Categories, **7-Class Emotion Probabilities**).
    *   **Thumbnails**: Retrieval now includes the `thumbnail` URL for UI display.
3.  **Dynamic Sorting**:
    *   **Context Check**: Determines if a `tone_filter` is active.
    *   **Ranking**: Applies either the Hybrid Scoring formula OR the Probability Sorting logic.
4.  **Ranked Response**:
    *   Truncates to the user-defined `top_k`.
    *   Returns a rich dictionary including the calculated `tone_prob` for UI visualization (e.g., "Mood Score: 0.95").

---

## **4. Configuration & Orchestration**

### **Hyperparameters (`params.yaml`)**
The behavior of the engine is fully tunable via DVC:
```yaml
inference:
  embedding_provider: huggingface
  model_name: sentence-transformers/all-MiniLM-L6-v2
  collection_name: books
  top_k: 10
  popularity_weight: 0.2
  
tone_analysis:
  target_emotions: ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
```

### **DVC Integration (`dvc.yaml`)**
The inference engine is integrated into the project's **Directed Acyclic Graph (DAG)** as `stage_05_prediction`. 
*   **Reproducibility**: Any change to `popularity_weight` or `model_name` triggers a re-run.
*   **Verifiability**: It produces a `results.txt` artifact in `artifacts/prediction/`.
