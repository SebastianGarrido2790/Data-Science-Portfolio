# Deployment & MLOps

Let’s proceed with **Deployment & MLOps** to operationalize the LogisticRegression model (version 2) for predicting employee attrition. We’ll follow the outlined goals (project_overview) to ensure reliability, scalability, security, and continuous feedback, integrating the model into HR workflows at UseC, a fictional company. The current date and time are 05:20 PM -04 on Thursday, May 15, 2025.

### Preparation for Deployment
**Objective**: Serialize the model and integrate version control for model iterations.

1. **Model Serialization**:
   - The LogisticRegression model (version 2) and the `StandardScaler` are already serialized in MLflow (`models:/LogisticRegression/2`) and `joblib` (`../../models/scaler.pkl`), respectively, from previous steps.
   - Verify that the MLflow tracking URI (`file:///C:/Users/.../mlruns`) contains the model artifacts and metadata.

2. **Version Control**:
   - The project is assumed to be under Git version control (`12_Employee_Attrition` repository).
   - Tag the current state of the repository with a semantic version (e.g., `v1.0.0`) to mark the initial deployment:
     ```
     git tag -a v1.0.0 -m "Initial deployment of LogisticRegression v2 model"
     git push origin v1.0.0
     ```

### Deployment Method
**Objective**: Choose a deployment method that aligns with business operations.

- **Business Context**: HR at UseC needs monthly predictions for employee attrition to plan retention strategies. Real-time predictions are not required since HR reviews predictions manually (as per Step 6 mitigation for false positives).
- **Deployment Choice**: Batch prediction is suitable, as it aligns with the monthly cadence. We’ll create a script to generate predictions on new employee data and integrate it into HR’s existing batch workflows (e.g., via a scheduled job).

#### Packaging & Versioning
**Objective**: Containerize the model, register it in MLflow, and integrate experimentation tracking.

1. **Containerization with Docker**:
   - Create a Docker image to package the prediction script, dependencies, and model artifacts for consistent deployment across environments.

### A single `Dockerfile` or separate ones?

Let’s decide whether to use a single `Dockerfile` or separate ones for the batch predictions and the FastAPI real-time API. Both approaches are viable, but the choice depends on how you want to manage and deploy these components at UseC. Since we're at 09:00 PM -04 on Thursday, May 15, 2025, let’s consider the operational needs and best practices for deployment.

#### Option 1: Separate Dockerfiles
- **Structure**:
  - `Dockerfile.batch`: For batch predictions (`predict_model.py`).
  - `Dockerfile.api`: For the FastAPI real-time API (`app.py`).
- **Pros**:
  - **Isolation**: Each component has its own image, reducing the risk of dependency conflicts (e.g., FastAPI/Uvicorn for API vs. batch script requirements).
  - **Clarity**: Separate images make it clear which is for batch processing and which is for real-time API, improving maintainability.
  - **Deployment Flexibility**: You can deploy batch predictions on a scheduled job (e.g., AWS Batch) and the API on a real-time service (e.g., AWS ECS) independently.
- **Cons**:
  - **Duplication**: Some setup steps (e.g., copying `requirements.txt`, `scaler.pkl`) are repeated.
  - **Maintenance Overhead**: Two Dockerfiles mean two images to build and manage.

#### Option 2: Single Dockerfile with Multi-Command Support
- **Structure**:
  - A single `Dockerfile` that builds one image but supports both batch predictions and API via different entrypoints or command-line arguments.
- **Pros**:
  - **Reduced Duplication**: One image to build and maintain, minimizing redundancy.
  - **Consistency**: Ensures both components use the same base environment and dependencies.
- **Cons**:
  - **Complexity**: Requires logic to switch between batch and API modes (e.g., environment variables or command-line arguments).
  - **Potential Overhead**: The image includes dependencies for both tasks, which might be unnecessary for one use case (e.g., batch doesn’t need Uvicorn).

#### Recommendation
- **Separate Dockerfiles** are recommended for this scenario:
  - **Reason**: Batch predictions (monthly HR workflow) and real-time API (on-demand predictions) serve distinct purposes with different deployment patterns. Isolating them ensures cleaner management, deployment, and scaling.
  - **Use Case Alignment**: HR currently uses batch predictions, and the API is a future enhancement. Keeping them separate allows HR to continue batch operations without adopting the API until ready.
  - **Scalability**: You can scale the API independently (e.g., multiple replicas in AWS ECS) without affecting batch jobs.

---

### Implementation: Separate Dockerfiles

#### 1. `Dockerfile.batch` (Batch Predictions)
This Dockerfile remains as-is for the batch prediction script (`predict_model.py`).

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY pyproject.toml uv.lock .
RUN uv sync

COPY src/models/predict_model.py .  
COPY models/scaler.pkl /app/models/scaler.pkl

CMD ["python", "predict_model.py"]
```

- **Build and Run**:
  ```
  docker build -f Dockerfile.batch -t employee-attrition-predictor-batch .
  docker run --rm -v $(pwd)/data:/app/data employee-attrition-predictor-batch
  ```

#### 2. `Dockerfile.api` (FastAPI Real-Time API)
This Dockerfile is for the FastAPI application (`app.py`).

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install uv
RUN pip install uv

COPY pyproject.toml uv.lock .
RUN uv sync

COPY app.py .  
COPY models/scaler.pkl /app/models/scaler.pkl

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- **Build and Run**:
  ```
  docker build -f Dockerfile.api -t employee-attrition-api .
  docker run -d -p 8000:8000 -v $(pwd)/models:/app/models employee-attrition-api
  ```

- **Explanation**:
  - Uses Python 3.9 slim base image for a lightweight container.
  - Installs dependencies from `requirements.txt` (assumed to include `pandas`, `numpy`, `scikit-learn`, `mlflow`, etc.).
  - Copies the prediction script and scaler.
  - Runs the prediction script as the default command.

2. **Requirements File**:
   - Create a `requirements.txt` to ensure consistent dependencies.

```plaintext
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
mlflow==2.12.2
xgboost==2.0.3
joblib==1.4.2
fastapi==0.110.0
uvicorn==0.29.0
```

- **Note**: Both Dockerfiles use the same `requirements.txt`. The batch image will install FastAPI/Uvicorn (unused), and the API image will install XGBoost (unused). This is a minor inefficiency but ensures consistency. If dependency conflicts arise later, you can split `requirements.txt` into `requirements-batch.txt` and `requirements-api.txt`.

3. **Register Model in MLflow**:
   - The model is already registered as `LogisticRegression` version 2 in MLflow. Confirm via the MLflow UI:
     ```
     mlflow ui --backend-store-uri file:///C:/Users/.../mlruns
     ```
   - Access the UI at `http://localhost:5000` to verify the model’s metadata (e.g., run ID, metrics: recall 0.8298, precision 0.21, F1 0.3319).

4. **Experimentation Tracking**:
   - MLflow already tracks experiments. Add a script to log deployment metadata (e.g., deployment date, environment) to the MLflow run.

#### Integration and Scalability
**Objective**: Integrate with HR systems and ensure scalability.

1. **Integration with HR Workflows**:
   - HR provides new employee data monthly in CSV format (e.g., `new_employees_may_2025.csv`).
   - The prediction script (`predict_model.py`) reads this CSV, generates predictions, and saves them to `data/predictions/predictions_may_2025.csv`.
   - HR accesses predictions via a shared directory and reviews them manually, as per Step 6 mitigation.
   - Schedule the batch job using a cron job (Linux) or Task Scheduler (Windows). Example cron job (runs monthly on the 1st at 8 AM):
     ```
     0 8 1 * * docker run --rm -v /path/to/data:/app/data employee-attrition-predictor
     ```

2. **Scalability**:
   - The dataset (1470 employees) is small, and batch predictions take seconds. Docker ensures the script runs consistently across environments.
   - For future scalability (e.g., if employee count grows to 100,000), consider:
     - Using a distributed batch processing framework like Apache Spark.
     - Hosting the model on a cloud service (e.g., AWS Batch) for parallel processing.

#### CI/CD Integration and Testing
**Objective**: Implement CI/CD practices and test the deployment.

1. **CI/CD Pipeline**:
   - Use GitHub Actions for CI/CD. Below is a workflow to build, test, and deploy the Docker image.

```yaml
name: Deploy Employee Attrition Model

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  build-and-test-batch:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build Docker image (Batch)
        run: docker build -f Dockerfile.batch -t employee-attrition-predictor-batch:latest .

      - name: Run tests (Batch)
        run: |
          docker run --rm -v $(pwd)/data:/app/data employee-attrition-predictor-batch
          if [ -f data/predictions/predictions.csv ]; then
            echo "Batch predictions generated successfully"
          else
            echo "Batch prediction failed" && exit 1
          fi

      - name: Push to Docker Hub (Batch)
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag employee-attrition-predictor-batch:latest ${{ secrets.DOCKER_USERNAME }}/employee-attrition-predictor-batch:v1.0.0
          docker push ${{ secrets.DOCKER_USERNAME }}/employee-attrition-predictor-batch:v1.0.0

  build-and-test-api:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Build Docker image (API)
        run: docker build -f Dockerfile.api -t employee-attrition-api:latest .

      - name: Run tests (API)
        run: |
          docker run -d -p 8000:8000 --name api-test employee-attrition-api
          sleep 5
          curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
          docker stop api-test

      - name: Push to Docker Hub (API)
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker tag employee-attrition-api:latest ${{ secrets.DOCKER_USERNAME }}/employee-attrition-api:v1.0.0
          docker push ${{ secrets.DOCKER_USERNAME }}/employee-attrition-api:v1.0.0
```

- **Explanation**:
  - Triggers on pushes or pull requests to the `main` branch.
  - Builds the Docker image, runs a test to ensure predictions are generated, and pushes the image to Docker Hub on `main` branch pushes.
  - Requires `DOCKER_USERNAME` and `DOCKER_PASSWORD` secrets in GitHub Actions.

2. **Testing**:
   - The test in the CI/CD pipeline verifies that predictions are generated (`predictions.csv` exists).
   - Manually test the Docker container locally:
     ```
     docker build -t employee-attrition-predictor .
     docker run --rm -v $(pwd)/data:/app/data employee-attrition-predictor
     ```
   - Check `data/predictions/predictions.csv` for expected output (775 attritions for LogisticRegression, 413 for XGBoost, as seen in Step 6).

#### Monitoring, Maintenance, and Retraining
**Objective**: Set up monitoring, schedule retraining, and implement rollback mechanisms.

1. **Monitoring**:
   - Log prediction statistics (e.g., number of attritions predicted) to MLflow.
   - Add monitoring to `predict_model.py` to log metrics like prediction distribution and runtime.
   - Use a monitoring tool like Prometheus to track system metrics (e.g., CPU, memory usage) if deployed on a server.
   - Detect concept drift by comparing prediction distributions over time (e.g., if attrition rate drops significantly, retrain the model).

2. **Periodic Retraining**:
   - Schedule retraining every 6 months or if HR reports significant changes in employee demographics or attrition patterns.
   - Use `train_model.py` to retrain, log the new model to MLflow, and update the version (e.g., `LogisticRegression` version 3).

3. **Rollback Mechanism**:
   - MLflow allows switching to a previous model version (e.g., `models:/LogisticRegression/1`) if performance degrades.
   - Update the `predict_model.py` script to allow specifying the model version as an environment variable (e.g., `MODEL_VERSION=1 python predict_model.py`).

#### Documentation and Knowledge Transfer
**Objective**: Document the deployment process and prepare a handover.

- **Deployment Documentation**:
  - **Setup**:
    - Clone the repository: `git clone <repo-url>`.
    - Build the Docker image: `docker build -t employee-attrition-predictor .`.
    - Run predictions: `docker run --rm -v /path/to/data:/app/data employee-attrition-predictor`.
  - **Dependencies**: See `requirements.txt`.
  - **Model Details**: LogisticRegression (version 2), recall 0.8298, precision 0.21, F1 0.3319.
  - **Monitoring**: Check MLflow UI for prediction logs; monitor prediction distribution for drift.
  - **Retraining**: Run `train_model.py` every 6 months or on drift detection.
  - **Rollback**: Update `MODEL_VERSION` environment variable to revert to a previous version.

- **Handover**:
  - Share documentation with HR and IT teams.
  - Conduct a training session on running the batch job, interpreting predictions, and accessing MLflow UI.

#### Post-Deployment Review
**Objective**: Capture lessons learned and ensure alignment with business goals.

- **Review**:
  - **Successes**: The model successfully deployed with batch predictions, meeting the recall goal (0.8298 > 0.70). Docker ensures consistency, and CI/CD automates updates.
  - **Challenges**: High false positive rate (775 predictions, precision 0.21) requires HR review, as planned. Scalability may become an issue with larger datasets.
  - **Lessons Learned**:
    - Aligning preprocessing between training and prediction (e.g., feature names, scaler scope) was critical and required multiple iterations.
    - MLflow URI handling on Windows needed careful formatting (`file:///C:/...`).
    - Batch deployment suits current needs, but real-time API deployment may be needed if HR requires faster predictions in the future.

- **Alignment with Business Goals**:
  - The deployment identifies most at-risk employees, supporting retention strategies.
  - HR review mitigates false positives, ensuring cost-effectiveness.
  - Monitoring and retraining plans ensure long-term reliability.

---

### Outcome
- The LogisticRegression model (version 2) is deployed as a batch prediction system, integrated into HR workflows at UseC.
- Docker ensures reliability, CI/CD enables automated updates, and MLflow provides tracking and versioning.
- Monitoring, retraining, and rollback mechanisms ensure maintainability, while documentation facilitates knowledge transfer.
- The system is scalable for current needs (1470 employees) but can be extended for larger datasets using cloud batch processing.

### Next Steps
- **Monitor Performance**: Track prediction distributions and HR feedback over the next 3 months to detect drift.
- **Optimize Threshold**: Experiment with a threshold of 0.45 for LogisticRegression to reduce false positives while maintaining recall above 0.70.
- **Plan for Future Enhancements**: Consider real-time API deployment (e.g., using Flask or FastAPI) if HR needs faster predictions.
