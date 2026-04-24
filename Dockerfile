# --- Stage 1: Build ---
FROM nvidia/cuda:12.1.0-base-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.7.1

RUN apt-get update && apt-get install -y python3-pip python3-dev
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock ./
# Install dependencies into a virtualenv in the current directory
RUN poetry config virtualenvs.in-project true && \
    poetry install --no-root --only main

# --- Stage 2: Final ---
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0

CMD ["streamlit", "run", "app.py"]