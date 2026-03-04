#!/bin/bash
echo "[entrypoint] Installing runtime dependencies..."
pip install -q pandas numpy google-genai 2>/dev/null
echo "[entrypoint] Done. Starting Django..."
exec python web/manage.py runserver 0.0.0.0:9001
