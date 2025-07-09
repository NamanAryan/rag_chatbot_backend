# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create cache directories with proper permissions
RUN mkdir -p /app/cache/huggingface /app/cache/sentence_transformers
RUN chmod -R 777 /app/cache

# Set environment variables for cache locations
ENV HF_HOME=/app/cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download AI models during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/cache/sentence_transformers')"

# Remove any lock files
RUN find /app/cache -name "*.lock" -delete 2>/dev/null || true

# Create upload directory
RUN mkdir -p /app/chroma_db_uploads

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
