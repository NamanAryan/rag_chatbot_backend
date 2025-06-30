# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9
FROM python:${PYTHON_VERSION}-slim as base

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Create cache directories with proper permissions for AI models
RUN mkdir -p /app/cache/huggingface /app/cache/sentence_transformers
RUN chmod -R 777 /app/cache

# Set environment variables for cache locations
ENV HF_HOME=/app/cache/huggingface
ENV SENTENCE_TRANSFORMERS_HOME=/app/cache/sentence_transformers
ENV TRANSFORMERS_CACHE=/app/cache/huggingface

# Create a non-privileged user that the app will run under.
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Download dependencies as a separate step to take advantage of Docker's caching.
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install -r requirements.txt

# Pre-download AI models during build to avoid runtime permission issues
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', cache_folder='/app/cache/sentence_transformers')"

# Remove any lock files that might have been created during model download
RUN find /app/cache -name "*.lock" -delete 2>/dev/null || true

# Create upload directory for your chatbot file uploads
RUN mkdir -p /app/chroma_db_uploads

# Switch to the non-privileged user to run the application.
USER appuser

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application with proper host and port binding
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
