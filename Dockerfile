# QNTI Trading System - Cloud Container
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for cloud deployment
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data logs

# Set environment variables for cloud deployment
ENV PYTHONPATH=/app
ENV FLASK_ENV=production
ENV QNTI_PORT=5002
ENV QNTI_CLOUD_MODE=true
ENV MT5_ENABLED=false

# Expose port
EXPOSE 5002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5002/health || exit 1

# Run the application
CMD ["python", "qnti_main_system.py"] 