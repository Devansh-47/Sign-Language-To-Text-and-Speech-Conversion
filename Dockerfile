FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pyenchant
RUN apt-get update && apt-get install -y \
    enchant-2 \
    libenchant-2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements-server.txt .
RUN pip install --no-cache-dir -r requirements-server.txt

# Copy application files
COPY landmark_server.py .
COPY app/ ./app/

# Expose port
EXPOSE 8000

# Run the server
CMD ["python", "landmark_server.py"]
