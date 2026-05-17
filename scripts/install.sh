#!/usr/bin/env bash
# Deprecated: just run devcolorbot from the project root.
set -euo pipefail
cd "$(dirname "$0")/.."
exec ./devcolorbot "$@"
