#!/bin/sh
set -e

export PYTHONPATH=/app

until pg_isready -h "${POSTGRES_HOST:-postgres}" -p "${POSTGRES_PORT:-5432}" -U "${POSTGRES_USER:-postgres}"; do
  echo "Waiting for postgres..."
  sleep 1
done

until alembic upgrade head; do
  echo "Waiting for database migrations to succeed..."
  sleep 1
done

uvicorn app.main:app --host 0.0.0.0 --port 8000
