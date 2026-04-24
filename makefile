# --- Configuration ---
PROJECT_NAME = vision-track
PYTHON_VERSION = 3.11
VENV = .venv

# OS Detection and Path Configuration
ifeq ($(OS),Windows_NT)
    PYTHON = python
    BIN = $(VENV)/Scripts
    ACTIVATE_CMD = $(BIN)\activate
    RM = rmdir /s /q
    # On Windows, we use 'python -m' to ensure we use the venv's tools
    RUN = $(BIN)\python -m
else
    PYTHON = python3
    BIN = $(VENV)/bin
    ACTIVATE_CMD = source $(BIN)/activate
    RM = rm -rf
    RUN = $(BIN)/python -m
endif

# --- Professional Workflow Targets ---

.PHONY: setup install boilerplate check-gpu clean help

## setup: Initialize environment, install poetry, and create project structure
setup: boilerplate
	@echo "Creating virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "Installing/Updating core tools..."
	$(BIN)/python -m pip install --upgrade pip poetry
	@echo "Installing dependencies via Poetry..."
	$(BIN)/poetry install
	@$(MAKE) check-gpu
	@echo "------------------------------------------------"
	@echo "✅ Setup Complete. To activate, run:"
	@echo "   $(ACTIVATE_CMD)"

## boilerplate: Create VisionTrack directory structure (Cross-Platform)
boilerplate:
	@echo "Generating VisionTrack structure..."
	@$(PYTHON) -c "import os; [os.makedirs(d, exist_ok=True) for d in ['data/raw_videos', 'data/raw_images', 'models/checkpoints', 'utils', 'reports/demo_results', 'logs']]"
	@$(PYTHON) -c "import os; [open(os.path.join(d, '__init__.py'), 'a').close() for d in ['models', 'utils']]"
	@if [ ! -f .gitignore ]; then \
		echo ".venv/\n__pycache__/\n*.pyc\n.DS_Store\ndata/\nmodels/checkpoints/*.pt\nlogs/" > .gitignore; \
	fi
	@echo "✅ Folders and .gitignore initialized."

## check-gpu: Verify if CUDA is available (Critical for your Gaming Laptop)
check-gpu:
	@echo "Checking Hardware Acceleration..."
	@$(BIN)/python -c "import torch; print('🚀 CUDA Available: ', torch.cuda.is_available()); print('🎮 Device Name: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Only')"

## run: Start the Streamlit VisionTrack Dashboard
run:
	$(BIN)/streamlit run app.py

## docker-build: Build the container for professional deployment
docker-build:
	docker build -t $(PROJECT_NAME):latest .

## clean: Remove environment and temporary files
clean:
	@$(RM) $(VENV)
	@$(PYTHON) -c "import shutil; [shutil.rmtree(p) for p in ['__pycache__', '.ipynb_checkpoints'] if os.path.exists(p)]"
	@echo "✅ Cleanup complete."

## help: Show this help message
help:
	@echo "VisionTrack Management Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'