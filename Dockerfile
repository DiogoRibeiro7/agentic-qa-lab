FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1         PYTHONUNBUFFERED=1         POETRY_VERSION=1.8.3

WORKDIR /app

RUN pip install --no-cache-dir "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false         && poetry install --only main --no-interaction --no-ansi

COPY src ./src
COPY apps ./apps
COPY README.md ./README.md

CMD ["python", "-m", "agentic_qa_lab.cli", "--help"]
