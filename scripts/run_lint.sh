#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip
pip install ruff==0.4.10

ruff check src scripts tests
