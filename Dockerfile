# --- Stage 1: Build Environment ---
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1

# Install system dependencies for build
RUN apt-get update && apt-get install -y python3-pip python3-dev curl
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Install ONLY production dependencies (skip dev tools like pytest)
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

# --- Stage 2: Runtime Runner ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# 1. Install CV2 System Dependencies (Critical for OpenCV/YOLO)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copy the pre-built environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 3. Copy Application Structure
# We explicitly copy these to ensure the image has the 'brains' (models) 
# and 'tools' (utils) alongside the app.
COPY models/ ./models/
COPY utils/ ./utils/
COPY app.py ./app.py
COPY README.md ./README.md

# 4. Create placeholders for data (to be mounted at runtime)
RUN mkdir -p data/raw_videos reports/demo_results logs

# Expose Streamlit port
EXPOSE 8501

# Start the dashboard
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]