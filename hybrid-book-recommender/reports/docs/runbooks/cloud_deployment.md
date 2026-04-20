# Cloud Deployment Runbook

## 1. Executive Summary
This document outlines the **Cloud Deployment Strategy** for the *Hybrid Book Recommender*. While the project is currently running in a local verification environment, the architecture is fully designed for **Continuous Deployment (CD)** to AWS EC2. The code is fully written and tested locally; to go live, you just need to add the API secrets.

This architecture demonstrates a production-grade MLOps pattern: **"Push-to-Deploy"**, where code committed to GitHub automatically updates the live application server without manual intervention.

## 2. Infrastructure Design (AWS EC2)
The target infrastructure is an **AWS EC2** instance, acting as a Docker Host.

*   **OS:** Ubuntu Server 24.04 LTS (Stable, widely supported).
*   **Instance Type:** `t3.micro` or `t2.micro` (Cost-effective for inference).
*   **Security:**
    *   **Port 22 (SSH):** Restricted to GitHub Actions Runner IPs (or specific admin IPs).
    *   **Port 7860 (App):** Open to the world (`0.0.0.0/0`) for user access.

## 3. The CD Pipeline
The deployment logic is embedded in `.github/workflows/main.yaml` handling the full lifecycle:

### **Stage 1: CI (Continuous Integration)**
*   **Trigger:** Push to `main`.
*   **Action:** Runs `pytest`.
*   **Outcome:** If tests pass, proceeds to Stage 2.

### **Stage 2: Artifact Creation**
*   **Action:** Builds the Docker image.
*   **Registry:** Pushes the image to **GitHub Container Registry (GHCR)**.
*   **Tagging:** Images are tagged as `latest` for immediate deployment.

### **Stage 3: CD (Continuous Deployment)**
*   **Action:** GitHub Actions establishes an SSH connection to the EC2 instance.
*   **Tool:** `appleboy/ssh-action`.
*   **Commands Executed on Server:**
    1.  `docker login ghcr.io` (Authenticated pull).
    2.  `docker pull ...` (Fetch new logic).
    3.  `docker stop ...` (Zero-downtime architecture would use Blue/Green, but Stop/Start is acceptable for this scale).
    4.  `docker run ...` (Restart service).

## 4. Configuration Requirements
To activate this pipeline on a live account, the following **GitHub Secrets** must be configured in the repository settings:

| Secret Name | Description |
| :--- | :--- |
| `EC2_HOST` | The Public IP address of the AWS instance (e.g., `54.x.x.x`). |
| `EC2_USERNAME` | The SSH user (default for Ubuntu keys is `ubuntu`). |
| `EC2_SSH_KEY` | The content of the private key file (`.pem`). |

## 5. Portfolio Verification
Although the lice cloud deployment is simulated in this version (due to resource constraints), the codebase contains all necessary components for a real-world deployment:
*   **Infrastructure Code:** documented in this report.
*   **Pipeline Code:** defined in `.github/workflows/main.yaml`.
*   **Containerization:** defined in `Dockerfile`.
*   **Helper Scripts:** `scripts/deploy.sh` serves as a manual deployment tool for admins SSHing directly into the server.

## 6. Deployment Scripts (`scripts/deploy.sh`)
For manual deployments or debugging, the `deploy.sh` script automates the Docker lifecycle on the EC2 instance. It serves as a mirroring logic to the CI/CD pipeline command block.

```bash
# Example Usage (On EC2)
./scripts/deploy.sh
```

This proves the ability to design and implement a complete **End-to-End MLOps Lifecycle** from data ingestion to cloud deployment.
