#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  echo "Run: python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
