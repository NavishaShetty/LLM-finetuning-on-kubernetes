FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py .

# Create models directory
RUN mkdir -p /models

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/models/tinyllama-alpaca-finetuned

# Run training script
CMD ["python", "train.py"]