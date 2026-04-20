#!/bin/bash

# Deployment Script for Hybrid Book Recommender
# Run this on the AWS EC2 instance

set -e  # Exit immediately if a command exits with a non-zero status

APP_NAME="hybrid-recommender"
# REPLACE WITH YOUR OWN IMAGE: ghcr.io/<your-username>/hybrid-book-recommender:latest
IMAGE_NAME="ghcr.io/sebastiangarrido2790/hybrid-book-recommender:latest"

echo "=========================================="
echo "Starting Deployment: $(date)"
echo "Target Image: $IMAGE_NAME"
echo "=========================================="

# 1. Login to GitHub Container Registry
# Assumes CR_PAT (Personal Access Token) is set in the environment or user is already logged in
# In CI/CD usage, we might pass the token or assume the runner is authenticated.
# For this script, we assume the specific `docker login` command happens via the SSH command in GitHub Actions,
# or we can ask for it here.
# To keep it simple for the SSH action, we usually rely on 'docker login' having happened *before* calling this,
# or we include it in the one-liner sent via SSH.

# 2. Pull the latest image
echo "d Pulling latest image..."
docker pull $IMAGE_NAME

# 3. Stop and Remove existing container (if running)
if [ "$(docker ps -q -f name=$APP_NAME)" ]; then
    echo "d Stopping existing container..."
    docker stop $APP_NAME
fi

if [ "$(docker ps -aq -f name=$APP_NAME)" ]; then
    echo "d Removing existing container..."
    docker rm $APP_NAME
fi

# 4. Run the new container
echo "d Starting new container..."
docker run -d \
    --name $APP_NAME \
    --restart always \
    -p 7860:7860 \
    $IMAGE_NAME

# 5. Cleanup unused images (optional, saves space)
echo "d Cleaning up old images..."
docker image prune -f

echo "=========================================="
echo "Deployment Successful!"
echo "=========================================="
