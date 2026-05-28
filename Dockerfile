# Use an official stable lightweight Python runtime as a parent base image
FROM python:3.10-slim

# Set system environment variables to optimize Python runtime behaviors
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Establish the functional isolation directory inside the container
WORKDIR /app

# Install native OS binary libraries required for compilation of rtree components
RUN apt-get update && apt-get install -y --no-install-recommends \
    libspatialindex-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the dependency manifest into the working directory container
COPY requirements.txt

# Upgrade pip and execute dependency installation via the cached layout
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the core Python application script into the workspace container
COPY main.py

# Explicitly expose port 8080 to match Cloud Run inbound traffic maps
EXPOSE 8080

# Launch the FastAPI Uvicorn application server on container startup
CMD ["python", "main.py"]
