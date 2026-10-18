FROM python:3.11-slim

# System deps for PyMuPDF (libmupdf) and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Install the package in editable mode
RUN pip install --no-cache-dir -e .

# Pre-download fastembed models so first request is instant
RUN python -c "\
from fastembed import TextEmbedding, SparseTextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5'); \
SparseTextEmbedding('Qdrant/bm25')"

# Pre-download cross-encoder reranker model (ONNX via fastembed — no PyTorch)
RUN python -c "\
from fastembed.rerank.cross_encoder import TextCrossEncoder; \
TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# Warm up local OCR (models ship inside the pip package; this just verifies the install)
RUN python -c "from rapidocr_onnxruntime import RapidOCR; RapidOCR()"

# Runtime data dirs (will be overridden by the bind mount in compose)
RUN mkdir -p /app/data/uploads /app/data/qdrant

EXPOSE 8000

CMD ["uvicorn", "notebook_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-dir", "src"]
