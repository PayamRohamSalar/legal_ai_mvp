#!/bin/bash

# This script creates the complete directory and file structure for the Legal AI Assistant project.

echo "Creating project structure for Legal AI Assistant..."

# Create the root directory
mkdir -p project_root
cd project_root

# Create top-level files
touch README.md
touch requirements.txt

# --- Configuration ---
echo "Creating config directory..."
mkdir -p config
touch config/config.py
touch config/llm_config.py
touch config/database_config.py
touch config/.env.example

# --- Documentation ---
echo "Creating docs directory..."
mkdir -p docs
touch docs/architecture.md
touch docs/technology_stack.md
touch docs/deployment_strategy.md

# --- Source Code (src) ---
echo "Creating src directory and sub-packages..."
mkdir -p src/api
mkdir -p src/core
mkdir -p src/data_processing
mkdir -p src/database
mkdir -p src/embeddings
mkdir -p src/llm_integration
mkdir -p src/query_processing
mkdir -p src/response_generation
mkdir -p src/search

# Add __init__.py to make them Python packages
touch src/__init__.py
touch src/api/__init__.py
touch src/core/__init__.py
touch src/data_processing/__init__.py
touch src/database/__init__.py
touch src/embeddings/__init__.py
touch src/llm_integration/__init__.py
touch src/query_processing/__init__.py
touch src/response_generation/__init__.py
touch src/search/__init__.py
touch src/api/main.py # Entry point for API

# --- Frontend Applications ---
echo "Creating frontend directories..."
# Streamlit MVP
mkdir -p streamlit_app
touch streamlit_app/app.py

# React Production Frontend
mkdir -p frontend/src/components
touch frontend/package.json

# --- Tests ---
echo "Creating tests directory..."
mkdir -p tests/unit_tests
mkdir -p tests/integration_tests
touch tests/__init__.py
touch tests/unit_tests/__init__.py
touch tests/integration_tests/__init__.py

# --- Deployment ---
echo "Creating deployment directory..."
mkdir -p deployment/scripts
mkdir -p deployment/kubernetes
touch deployment/scripts/deploy.sh
touch deployment/kubernetes/deployment.yaml

# --- DevOps (CI/CD and Docker) ---
echo "Creating CI/CD and Docker configurations..."
# GitHub Actions
mkdir -p .github/workflows
touch .github/workflows/ci.yml

# Docker
mkdir -p docker
touch docker/Dockerfile
touch docker/docker-compose.yml

echo "Project structure created successfully inside 'project_root' directory."
cd ..