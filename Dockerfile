# One image for every Python service; the command selects the service (see docker-compose.yml).
# Stage 1: build the web app so the runtime image can serve it.
FROM node:22-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --silent
COPY web/ ./
RUN npm run build

# Stage 2: Python services.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 FALCON_ROOT=/app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
# Dependencies first, from pyproject.toml alone, so a change under src/ rebuilds in seconds instead of re-installing every
# package (FAILURE_DIARY #45: every rebuild of the day took 3-5 minutes for one edited file).
RUN python -c "import tomllib; print(chr(10).join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))" > /tmp/requirements.txt     && pip install --quiet -r /tmp/requirements.txt pytest pytest-asyncio
COPY src ./src
# Editable install without dependencies: the code runs from /app/src, so paths derived from __file__ (data/, web/dist, db/) resolve to /app.
RUN pip install --quiet --no-deps -e .
COPY db ./db
COPY scripts ./scripts
COPY eval ./eval
COPY data ./data
COPY --from=web /web/dist ./web/dist
EXPOSE 8000 8001 8002 8003 8010
CMD ["uvicorn", "falcon.services.runtime.app:app", "--host", "0.0.0.0", "--port", "8000"]
