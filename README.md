# Create a new project folder
mkdir <project_folder_name>

# Move into the project folder
cd <project_folder_name>

# Open the folder in VS Code
code .

# Create a new Conda environment with Python 3.10
conda create -p <env_name> python=3.10 -y

# Activate the environment (use full path to the environment)
conda activate <path_of_the_env>

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Initialize Git
git init

# Stage all files
git add .

# Commit changes
git commit -m "<write your commit message>"

# Push to remote (after adding remote origin)
git push


An intelligent document platform enabling users to upload, analyze, compare, and chat with PDFs using grounded multi-document reasoning.

🔍 Key Features
RAG-based document understanding using FAISS vector store
Multi-document comparison and contextual Q&A
Async FastAPI backend with:
Structured logging
Custom exception handling
CI/CD pipeline using GitHub Actions
Fully containerized and deployed on AWS ECS (Fargate)
🏗️ Architecture Highlights
Secure secrets management with AWS Secrets Manager
Load balancing via Application Load Balancer (ALB)
Least-privilege IAM roles for secure access
Streamlit-based interactive UI
🧰 Tech Stack

Python · FastAPI · Docker · LLMs · RAG · FAISS · AWS ECS (Fargate) · Amazon ECR · ALB · AWS Secrets Manager · CloudWatch · GitHub Actions · Streamlit

