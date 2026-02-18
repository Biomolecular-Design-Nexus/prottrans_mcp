FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    git gcc g++ wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install --ignore-installed fastmcp
RUN pip install --no-cache-dir --prefix=/install tiktoken

FROM python:3.10-slim AS runtime

WORKDIR /app
COPY --from=builder /install /usr/local
COPY src/ ./src/
COPY scripts/ ./scripts/
RUN mkdir -p tmp/inputs tmp/outputs

ENV PYTHONPATH=/app

CMD ["python", "src/server.py"]
