#!/bin/sh
set -e

# Start nginx in background
nginx -g "daemon off;" &

# Start FastAPI
exec uvicorn main:app --host 0.0.0.0 --port 8000
