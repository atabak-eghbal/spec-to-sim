#!/usr/bin/env bash
# Upload this repo's current commit to a Hugging Face Docker Space.
#   ./deploy/huggingface/deploy.sh <hf-username>/<space-name>
# Needs a one-time login first:  uvx --from huggingface_hub hf auth login
# Only files committed to git are uploaded (no .env, no runs/), plus a Space README header.
set -euo pipefail
SPACE="${1:?usage: deploy.sh <hf-username>/<space-name>}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE"
mkdir -p "$STAGE/.cache" && cp -R "$ROOT/.cache/physics" "$STAGE/.cache/physics"   # pre-built web facts (not in git)
{ cat <<'HDR'
---
title: Spec-to-Sim
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: One sentence to a checked MuJoCo robot simulation
---

HDR
  cat "$STAGE/README.md"; } > "$STAGE/README.tmp" && mv "$STAGE/README.tmp" "$STAGE/README.md"
uvx --from huggingface_hub hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
echo "Uploaded. Build logs: https://huggingface.co/spaces/$SPACE?logs=build"
