# Use a lightweight Python base image
FROM python:3.12-slim

# Install the required C++ graphics libraries for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy only the dependency files first (makes future updates faster)
COPY pyproject.toml poetry.lock ./

# Tell Poetry to install directly into the system, not a virtual environment
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root

# Copy your actual code and ONNX models
COPY . .

# Open the Streamlit port
EXPOSE 8501

# Launch the Command Center
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false"]