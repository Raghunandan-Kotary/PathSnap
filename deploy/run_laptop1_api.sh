#!/usr/bin/env bash
echo "========================================================"
echo "  Starting PathSnap Model Server on Laptop 1 (Port 8000)"
echo "========================================================"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
