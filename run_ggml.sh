#!/usr/bin/env sh
set -eu
exec python3 "$(dirname "$0")/cpp_ggml/scripts/run_ggml.py" "$@"
