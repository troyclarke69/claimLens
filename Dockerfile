# ClaimLens service image.
#
#   CPU demo (simulated predictor, runs anywhere):
#     docker build -t claimlens .
#     docker run -p 8000:8000 -v "%cd%/data:/app/data" claimlens            (Windows cmd)
#     docker run -p 8000:8000 -v "$(pwd)/data:/app/data" claimlens          (macOS/Linux)
#     then open http://localhost:8000/docs
#
#   GPU (real model): build with the CUDA base instead and serve the registry's production model:
#     docker build --build-arg BASE=pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime --build-arg GPU=1 -t claimlens-gpu .
#     docker run --gpus all -p 8000:8000 -e CLAIMLENS_PREDICTOR=hf \
#       -v "$(pwd)/adapters:/app/adapters" -v "$(pwd)/models:/app/models" claimlens-gpu
ARG BASE=python:3.11-slim
FROM ${BASE}
ARG GPU=0

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# fonts so the generator produces the same documents as on Kaggle
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-service.txt requirements-gpu.txt ./
RUN pip install -r requirements.txt -r requirements-service.txt && \
    if [ "$GPU" = "1" ]; then pip install -r requirements-gpu.txt; fi

COPY claimlens ./claimlens

# run as a non-root user
RUN useradd --create-home app && mkdir -p /app/runs /app/data /app/models /app/adapters && chown -R app /app
USER app

ENV CLAIMLENS_PREDICTOR=simulated CLAIMLENS_DATA=/app/data CLAIMLENS_AUDIT_LOG=/app/runs/service/audit.jsonl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/v1/health')"
CMD ["uvicorn", "claimlens.service:app", "--host", "0.0.0.0", "--port", "8000"]
