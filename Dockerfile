FROM python:3.11-slim

WORKDIR /app

# install deps first so Docker caches this layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy everything else
COPY . .

# the model gets downloaded on first run and cached here
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# pre-download the model during build so startup is faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 80

# gunicorn with a generous timeout because the first request
# triggers embedding computation which takes a while
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--timeout", "300", "--workers", "1", "app:app"]
