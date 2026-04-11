FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port (Hugging Face Spaces uses 7860)
EXPOSE 7860

# Health check — openenv-core registers /health with {"status": "healthy"}
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Run via the openenv-core server entry point
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
