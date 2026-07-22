FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed for compilation and ML libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip and install requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --default-timeout=100 --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure python finds the backend module
ENV PYTHONPATH=/app

CMD ["python", "backend/backend_daemon.py"]
