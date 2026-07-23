# Multi-stage Docker build for FastAPI application
# This creates a minimal, production-ready container

# Stage 1: Base image with Python
FROM python:3.11-slim as base

# Set working directory in container
WORKDIR /app

# Set environment variables
# PYTHONUNBUFFERED: Prevents Python from buffering stdout/stderr
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# This is kept minimal for a smaller image
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies
FROM base as dependencies

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir: Don't cache packages (smaller image)
# --upgrade: Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: Final production image
FROM base as production

# Copy installed dependencies from dependencies stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=dependencies /usr/local/bin /usr/local/bin

# Copy application code
COPY ./app /app/app

# Create a non-root user for security
# Running as non-root is a security best practice
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port 8000
# This is the port that uvicorn will listen on
EXPOSE 8000

# Health check
# This allows Docker/Kubernetes to verify the container is healthy
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

# Command to run the application
# Using uvicorn as the ASGI server
# --host 0.0.0.0: Listen on all network interfaces
# --port 8000: Listen on port 8000
# --workers 1: Number of worker processes (adjust based on CPU cores)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
