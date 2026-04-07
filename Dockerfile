FROM python:3.11-slim

WORKDIR /app

# Install build tools needed for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies first (cached layer)
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefer-binary \
        "fastapi>=0.115.0" \
        "pydantic>=2.0.0" \
        "uvicorn>=0.24.0" \
        "requests>=2.31.0" \
        "openenv-core[core]>=0.2.1"

# Copy application code
COPY releaseops_env/ /app/releaseops_env/
COPY server/ /app/server/
COPY tasks/ /app/tasks/
COPY baseline/ /app/baseline/
COPY data/ /app/data/
COPY openenv.yaml /app/openenv.yaml

# Install the local package (after source is present)
RUN pip install --no-cache-dir --no-deps -e .

ENV PYTHONUNBUFFERED=1
ENV PORT=7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
