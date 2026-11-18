#!/usr/bin/env bash
echo "========================================================"
echo "  Starting PathSnap Evaluator Dashboard on Laptop 2"
echo "========================================================"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$DIR"
source .venv/bin/activate
streamlit run ui/app.py
