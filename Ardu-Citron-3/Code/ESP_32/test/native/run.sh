#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 test/native/run_native.py
