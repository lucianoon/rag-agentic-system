FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py add_documents.py ./
COPY src ./src
COPY config ./config

# Nada roda como root; corpus e índice persistem no volume montado em /app/data
RUN useradd --create-home appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

# Modos de uso:
#   interativo:   docker compose run --rm rag-agent
#   tarefa única: docker compose run --rm rag-agent --task "sua pergunta"
ENTRYPOINT ["python", "main.py"]
