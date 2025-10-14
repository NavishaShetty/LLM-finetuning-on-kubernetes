FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install only essential packages one at a time
RUN pip install --no-cache-dir transformers==4.35.0
RUN pip install --no-cache-dir peft==0.7.0
RUN pip install --no-cache-dir accelerate==0.24.0
RUN pip install --no-cache-dir bitsandbytes==0.41.0
RUN pip install --no-cache-dir datasets==2.14.0
RUN pip install --no-cache-dir scipy

COPY train.py .
RUN mkdir -p /models

ENV PYTHONUNBUFFERED=1
ENV OUTPUT_DIR=/models/tinyllama-alpaca-finetuned

CMD ["python", "train.py"]