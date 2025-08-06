import pathlib

def create_project_structure():
    """
    Creates the complete directory and file structure for the Legal AI Assistant project.
    """
    print("Creating project structure for Legal AI Assistant using Python...")

    # Define the root directory for the project
    print("Creating project structure for Legal AI Assistant using Python...")
    project_root = pathlib.Path(r'D:\OneDrive\My Projects\Smart Legal Assistant\legal_ai_mvp')

    # Define all directories to be created
    directories = [
        # Top-level directories
        project_root / 'config',
        project_root / 'docs',
        project_root / 'deployment/scripts',
        project_root / 'deployment/kubernetes',
        project_root / '.github/workflows',
        project_root / 'docker',
        project_root / 'streamlit_app',
        project_root / 'frontend/src/components',
        
        # Source directories
        project_root / 'src/api',
        project_root / 'src/core',
        project_root / 'src/data_processing',
        project_root / 'src/database',
        project_root / 'src/embeddings',
        project_root / 'src/llm_integration',
        project_root / 'src/query_processing',
        project_root / 'src/response_generation',
        project_root / 'src/search',
        
        # Test directories
        project_root / 'tests/unit_tests',
        project_root / 'tests/integration_tests',
    ]

    # Define all empty files to be created
    files = [
        # Top-level files
        project_root / 'README.md',
        project_root / 'requirements.txt',
        
        # Configuration files
        project_root / 'config/config.py',
        project_root / 'config/llm_config.py',
        project_root / 'config/database_config.py',
        project_root / 'config/.env.example',

        # Documentation files
        project_root / 'docs/architecture.md',
        project_root / 'docs/technology_stack.md',
        project_root / 'docs/deployment_strategy.md',

        # Deployment and DevOps files
        project_root / 'deployment/scripts/deploy.sh',
        project_root / 'deployment/kubernetes/deployment.yaml',
        project_root / '.github/workflows/ci.yml',
        project_root / 'docker/Dockerfile',
        project_root / 'docker/docker-compose.yml',

        # Application files
        project_root / 'streamlit_app/app.py',
        project_root / 'frontend/package.json',
        project_root / 'src/api/main.py',

        # __init__.py files to mark directories as Python packages
        project_root / 'src/__init__.py',
        project_root / 'src/api/__init__.py',
        project_root / 'src/core/__init__.py',
        project_root / 'src/data_processing/__init__.py',
        project_root / 'src/database/__init__.py',
        project_root / 'src/embeddings/__init__.py',
        project_root / 'src/llm_integration/__init__.py',
        project_root / 'src/query_processing/__init__.py',
        project_root / 'src/response_generation/__init__.py',
        project_root / 'src/search/__init__.py',
        project_root / 'tests/__init__.py',
        project_root / 'tests/unit_tests/__init__.py',
        project_root / 'tests/integration_tests/__init__.py',
    ]

    # Create all directories
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {path}")

    # Create all files
    for path in files:
        path.touch(exist_ok=True)
        print(f"Created file: {path}")

    print("\nProject structure created successfully inside 'project_root' directory.")


if __name__ == "__main__":
    create_project_structure()