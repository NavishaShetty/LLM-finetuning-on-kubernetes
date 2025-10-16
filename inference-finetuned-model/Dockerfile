FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 8000

# Environment variables (can be overridden in Kubernetes)
ENV PYTHONUNBUFFERED=1
ENV MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
ENV FINETUNED_MODEL=shettynavisha25/tinyllama-alpaca-finetuned

# Run the app
CMD ["python", "app.py"]