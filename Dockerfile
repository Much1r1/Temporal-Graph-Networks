FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements or install python packages directly
COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch numpy pandas scikit-learn pyyaml pytest fastapi httpx uvicorn

ENV PYTHONPATH=/app

# Default command starts inference server
CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
