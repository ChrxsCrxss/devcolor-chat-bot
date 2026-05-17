#!/usr/bin/env bash
# Capture sample_output.txt for the three required assignment queries.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

if [[ ! -x "$ROOT/devcolorbot" ]]; then
  chmod +x "$ROOT/devcolorbot"
fi

OUT="$ROOT/sample_output.txt"
echo "/dev/color RAG — Sample Output" > "$OUT"
echo "Generated: $(date -u)" >> "$OUT"
echo "Repo: devcolor-chat-bot" >> "$OUT"
echo "========================================" >> "$OUT"

QUERIES=(
  "How can /dev/color help me develop my career?"
  "How can I contribute to /dev/color?"
  "In which cities is /dev/color located?"
)

ECHO_FLAG=""
if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama not detected — using --echo mode" | tee -a "$OUT"
  ECHO_FLAG="--echo"
fi

for q in "${QUERIES[@]}"; do
  echo "" >> "$OUT"
  echo "----------------------------------------" >> "$OUT"
  echo "USER: $q" >> "$OUT"
  echo "----------------------------------------" >> "$OUT"
  ./devcolorbot --once "$q" $ECHO_FLAG --no-color 2>&1 >> "$OUT" || true
done

echo "" >> "$OUT"
echo "Done. Output written to $OUT"
