FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

# requirements.txt pins a CUDA-specific torch build (+cu130) for local GPU
# dev machines. Plain `pip install torch` on Linux (even without that
# suffix) still pulls in ~1.5GB of bundled nvidia-cu13* runtime packages as
# dependencies -- installing from the dedicated /whl/cpu index avoids that
# entirely, which matters a lot on a memory/disk-constrained free host.
RUN grep -v -e extra-index-url -e "^torch==" requirements.txt > requirements.docker.txt \
    && pip install --no-cache-dir -r requirements.docker.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.12.0

COPY . .

ENV PORT=7860
EXPOSE 7860

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT}"]
