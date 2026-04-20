## Step 7.2: Deploy to AWS ECS with CI/CD

Let’s implement the deployment of Employee Attrition Prediction API to AWS ECS with CI/CD integration. We’ll modify the GitHub Actions workflow for deployment to AWS ECS, and provide instructions for setting up AWS resources.

#### AWS Setup Overview
We’ll deploy the API to AWS ECS (Elastic Container Service) using an ECS cluster with Fargate (serverless compute). We’ll set up:
- **ECR (Elastic Container Registry)**: To store the Docker image.
- **ECS Cluster and Task Definition**: To run the container.
- **Application Load Balancer (ALB)**: To handle HTTPS traffic.
- **GitHub Actions**: For CI/CD to build, push, and deploy the image.

#### Step 1: AWS Prerequisites
1. **Create an ECR Repository**:
   - Go to AWS Console → ECR → Create Repository.
   - Name: `employee-attrition-api`.
   - Note the repository URI (e.g., `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/employee-attrition-api`).

2. **Set Up an ECS Cluster**:
   - Go to ECS → Clusters → Create Cluster.
   - Cluster Name: `employee-attrition-cluster`.
   - Infrastructure: AWS Fargate (serverless).
   - Create.

3. **Create a Task Definition**:
   - Go to ECS → Task Definitions → Create new task definition.
   - Name: `employee-attrition-task`.
   - Container Details:
     - Image: `<aws_account_id>.dkr.ecr.<region>.amazonaws.com/employee-attrition-api:latest`.
     - Port Mappings: 8000 (for Uvicorn).
     - Environment Variables:
       - `API_KEY`: `your-secure-key`.
   - CPU: 256, Memory: 512 (adjust as needed).
   - Create.

4. **Set Up an Application Load Balancer (ALB)**:
   - Go to EC2 → Load Balancers → Create Load Balancer → Application Load Balancer.
   - Name: `employee-attrition-alb`.
   - Listeners: HTTP (port 80) and HTTPS (port 443) with an SSL certificate (use AWS Certificate Manager).
   - Target Group:
     - Name: `employee-attrition-tg`.
     - Port: 8000.
     - Health Check: `/` (returns HTML from `index.html`).
   - Create.

5. **Create an ECS Service**:
   - Go to ECS → Clusters → `employee-attrition-cluster` → Create Service.
   - Task Definition: `employee-attrition-task`.
   - Service Name: `employee-attrition-service`.
   - Desired Tasks: 1.
   - Load Balancer: Attach to `employee-attrition-alb` and the target group `employee-attrition-tg`.
   - Create.

6. **IAM Roles**:
   - Ensure the ECS task execution role (`ecsTaskExecutionRole`) has permissions to pull from ECR and access logs.
   - Add a GitHub Actions IAM user with permissions for:
     - `ecr:PutImage`, `ecr:InitiateLayerUpload`, etc., to push to ECR.
     - `ecs:UpdateService` to redeploy the service.

#### Step 2: Update GitHub Actions Workflow

Update `.github/workflows/deploy.yml` to build, push the image to ECR, and redeploy the ECS service.

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

      - name: Log in to Docker Hub
        if: github.ref == 'refs/heads/main'
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin

      - name: Push to Docker Hub (Batch)
        if: github.ref == 'refs/heads/main'
        run: |
          docker tag employee-attrition-predictor-batch:latest ${{ secrets.DOCKER_USERNAME }}/employee-attrition-predictor-batch:v1.0.0
          docker push ${{ secrets.DOCKER_USERNAME }}/employee-attrition-predictor-batch:v1.0.0

  build-and-deploy-api:
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
          docker run -d -p 8000:8000 --name api-test -e API_KEY=your-secret-key employee-attrition-api
          sleep 5
          curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -H "Authorization: Bearer your-secret-key" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
          docker stop api-test

      - name: Log in to AWS ECR
        if: github.ref == 'refs/heads/main'
        uses: aws-actions/amazon-ecr-login@v1
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: ${{ secrets.AWS_REGION }}

      - name: Tag and Push to AWS ECR
        if: github.ref == 'refs/heads/main'
        env:
          ECR_REGISTRY: ${{ secrets.AWS_ACCOUNT_ID }}.dkr.ecr.${{ secrets.AWS_REGION }}.amazonaws.com
          ECR_REPOSITORY: employee-attrition-api
        run: |
          docker tag employee-attrition-api:latest $ECR_REGISTRY/$ECR_REPOSITORY:latest
          docker push $ECR_REGISTRY/$ECR_REPOSITORY:latest

      - name: Deploy to AWS ECS
        if: github.ref == 'refs/heads/main'
        run: |
          aws ecs update-service --cluster employee-attrition-cluster --service employee-attrition-service --force-new-deployment
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          AWS_DEFAULT_REGION: ${{ secrets.AWS_REGION }}
```

- **Changes**:
  - Renamed `build-and-test-api` job to `build-and-deploy-api`.
  - Added API key to the test step (`-e API_KEY=your-secret-key` and `-H "Authorization: Bearer your-secret-key"`).
  - Added steps to log in to AWS ECR, push the image, and redeploy the ECS service.
  - Removed Docker Hub push for the API (replaced with ECR).

#### Step 3: Configure GitHub Secrets
Add the following secrets in your GitHub repository (Settings → Secrets and variables → Actions):
- `AWS_ACCESS_KEY_ID`: Your AWS IAM user’s access key.
- `AWS_SECRET_ACCESS_KEY`: Your AWS IAM user’s secret key.
- `AWS_REGION`: Your AWS region (e.g., `us-east-1`).
- `AWS_ACCOUNT_ID`: Your AWS account ID.

#### Step 4: Test the Deployment
1. **Push to `main` Branch**:
   - Commit the updated `app.py` and `.github/workflows/deploy.yml`.
   - Push to the `main` branch to trigger the GitHub Actions workflow.
   - Monitor the workflow in the GitHub Actions tab.

2. **Access the Deployed API**:
   - Once the ECS service is updated, get the ALB DNS name from the AWS Console (EC2 → Load Balancers).
   - Access the API at `http://<alb-dns-name>` (or `https` if configured).
   - Test the `/predict` endpoint with the API key:
     ```
     curl -X POST "http://<alb-dns-name>/predict" -H "Content-Type: application/json" -H "Authorization: Bearer your-secure-key" -d '{"Age": 35, "DailyRate": 1000, "DistanceFromHome": 5, "Education": 3, "EmployeeNumber": 1001, "EnvironmentSatisfaction": 3, "HourlyRate": 50, "JobInvolvement": 3, "JobLevel": 2, "JobSatisfaction": 4, "MonthlyIncome": 5000, "MonthlyRate": 12000, "NumCompaniesWorked": 2, "PercentSalaryHike": 10, "PerformanceRating": 3, "RelationshipSatisfaction": 3, "StockOptionLevel": 1, "TotalWorkingYears": 10, "TrainingTimesLastYear": 2, "WorkLifeBalance": 3, "YearsAtCompany": 5, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 2, "YearsWithCurrManager": 2, "BusinessTravel": "Travel_Rarely", "Department": "Research_&_Development", "EducationField": "Life_Sciences", "Gender": "Male", "JobRole": "Research_Scientist", "MaritalStatus": "Single", "OverTime": "No"}'
     ```

---

### 3. Next Steps
- **Monitoring**: Add Prometheus and Grafana for API uptime and latency monitoring on AWS ECS.
- **HTTPS**: Ensure the ALB uses HTTPS with a proper SSL certificate.
- **Refine Security**: Rotate API keys periodically and consider OAuth for more robust authentication.

---

## Why deploy to AWS?

Deploying the Employee Attrition Prediction API to AWS, specifically using AWS ECS, offers several compelling reasons that align with scalability, reliability, and operational efficiency for our project. Here’s a concise breakdown of why AWS is a suitable choice, tailored to our context:

### 1. **Scalability**
- **Elastic Container Service (ECS) with Fargate**: AWS ECS allows us to run containers in a serverless manner with Fargate, automatically scaling the number of tasks based on demand. This is ideal for HR use cases where prediction requests might spike during planning periods.
- **Global Infrastructure**: AWS provides a global network of data centers, enabling us to scale across regions if needed, ensuring low latency for users worldwide.

### 2. **Cost Efficiency**
- **Pay-as-You-Go Model**: With Fargate, you only pay for the compute resources your containers use, avoiding the overhead of managing physical servers. This is cost-effective for a project starting with low traffic, like our attrition prediction API.
- **Free Tier**: AWS offers a free tier for new users, which can offset initial costs for testing and deployment.

### 3. **Reliability and High Availability**
- **Built-in Redundancy**: AWS ECS integrates with Application Load Balancers (ALB) to distribute traffic across multiple instances, ensuring high availability and fault tolerance.
- **Managed Services**: AWS handles infrastructure maintenance, updates, and security patches, reducing your operational burden.

### 4. **Integration with CI/CD**
- **GitHub Actions Compatibility**: AWS integrates seamlessly with GitHub Actions, as demonstrated in our `deploy.yml` workflow. This allows automated builds, tests, and deployments to ECR and ECS, streamlining our development pipeline.
- **Container Support**: AWS ECR and ECS are designed for Docker containers, aligning with our existing `Dockerfile.api` setup.

### 5. **Security Features**
- **IAM and API Key Management**: AWS Identity and Access Management (IAM) secures access to ECS and ECR, while our implemented API key authentication can leverage AWS Secrets Manager for secure storage and rotation.
- **HTTPS with ALB**: The ALB supports SSL/TLS, ensuring encrypted communication, which is critical for handling sensitive HR data.

### 6. **Ecosystem and Tools**
- **Monitoring and Logging**: AWS provides CloudWatch for monitoring API performance and logs, which you can extend with Prometheus and Grafana as planned.
- **Developer Tools**: AWS offers services like CodePipeline and CodeDeploy, enhancing CI/CD capabilities if you expand beyond GitHub Actions.

### 7. **Industry Adoption and Support**
- **Widely Used**: AWS is the leading cloud provider, with extensive documentation, community support, and third-party integrations, making it easier to troubleshoot and scale your project.
- **HR Use Case Fit**: Many enterprises, including those in HR analytics, rely on AWS for its robust infrastructure, making it a trusted platform for our attrition prediction API.

### Why Not Other Options?
- **Google Cloud Run**: While scalable and serverless, it has less mature ECS-like container orchestration and tighter integration with Docker Hub compared to AWS ECR/ECS.
- **Azure Container Instances**: Offers similar serverless container support but has a steeper learning curve for ECS-specific workflows and less seamless GitHub Actions integration out of the box.
- **Local Hosting**: Running on a personal server lacks scalability, security, and reliability compared to AWS’s managed services.

### Conclusion
Deploying to AWS ECS leverages its scalability, cost efficiency, and robust ecosystem, making it an ideal choice to support our real-time HR prediction API as it grows. The integration with our existing Docker setup and CI/CD pipeline via GitHub Actions further simplifies the process, ensuring a smooth transition from local development to production.
