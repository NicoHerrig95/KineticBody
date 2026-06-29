#!/usr/bin/env bash
set -e

LOCAL_INPUT_PATH="$1"
LOCAL_OUTPUT_PATH="$2"

if [ -z "$LOCAL_INPUT_PATH" ] || [ -z "$LOCAL_OUTPUT_PATH" ]; then
  echo "Usage: $0 <local-input-video-path> <local-output-json-path>"
  exit 1
fi

INPUT_DIR="$(cd "$(dirname "$LOCAL_INPUT_PATH")" && pwd)"
INPUT_FILE="$(basename "$LOCAL_INPUT_PATH")"

OUTPUT_DIR="$(mkdir -p "$(dirname "$LOCAL_OUTPUT_PATH")" && cd "$(dirname "$LOCAL_OUTPUT_PATH")" && pwd)"
OUTPUT_FILE="$(basename "$LOCAL_OUTPUT_PATH")"

docker run --rm \
  --platform linux/amd64 \
  -v "$INPUT_DIR:/data:ro" \
  -v "$OUTPUT_DIR:/outputs" \
  kineticbody:latest \
  --input-path "/data/$INPUT_FILE" \
  --output-path "/outputs/$OUTPUT_FILE"