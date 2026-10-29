#!/usr/bin/env bash
# Upload this repo's current commit to a Hugging Face Space (Gradio SDK, free CPU Basic).
#   ./deploy/huggingface/deploy.sh <hf-username>/<space-name>          # upload
#   ./deploy/huggingface/deploy.sh --stage-only <dir>                   # just build the folder (for testing)
# Needs a one-time login first:  uvx --from huggingface_hub hf auth login
# Only files committed to git are uploaded (no .env, no runs/), plus the Franka model, the shared
# web-facts cache, the Space entrypoint, requirements and a Space README header.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [ "${1:-}" = "--stage-only" ]; then STAGE="${2:?dir}"; mkdir -p "$STAGE"; SPACE=""; else
  SPACE="${1:?usage: deploy.sh <hf-username>/<space-name>}"; STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT; fi
git -C "$ROOT" archive HEAD | tar -x -C "$STAGE"
mkdir -p "$STAGE/third_party/menagerie"
# Hugging Face refuses files under '.cache/': ship the web facts as kb_physics/; space_app.py copies
# them into .cache/physics/ at start-up, where the grounder reads them.
cp -R "$ROOT/.cache/physics" "$STAGE/kb_physics"
cp -R "$ROOT/third_party/menagerie/franka_emika_panda" "$STAGE/third_party/menagerie/"
cp "$ROOT/deploy/huggingface/space_app.py" "$STAGE/space_app.py"
# Loose bounds from pyproject.toml (the lockfile's exact pins clash with Gradio's own), except MuJoCo,
# pinned to the locked version so physics and the restitution calibration match what was tested.
MJ="$(cd "$ROOT" && uv export --no-dev --no-hashes --format requirements-txt | grep -E '^mujoco==' | cut -d' ' -f1)"
python3 -c "import tomllib,sys; d=tomllib.load(open('$ROOT/pyproject.toml','rb'))['project']['dependencies']; print('\n'.join(x for x in d if not x.startswith('mujoco')))" > "$STAGE/requirements.txt"
echo "$MJ" >> "$STAGE/requirements.txt"
printf 'libosmesa6\nlibgl1\nlibglib2.0-0\n' > "$STAGE/packages.txt"
rm -f "$STAGE/Dockerfile" "$STAGE/.dockerignore"   # a Gradio Space must not look like a Docker one
{ cat <<'HDR'
---
title: Spec-to-Sim
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.1
python_version: "3.11"
app_file: space_app.py
pinned: false
license: mit
short_description: One sentence to a checked MuJoCo robot simulation
---

HDR
  cat "$STAGE/README.md"; } > "$STAGE/README.tmp" && mv "$STAGE/README.tmp" "$STAGE/README.md"
[ -z "$SPACE" ] && { echo "Staged in $STAGE"; exit 0; }
uvx --from huggingface_hub hf upload "$SPACE" "$STAGE" . --repo-type space --commit-message "Deploy $(git -C "$ROOT" rev-parse --short HEAD)"
echo "Uploaded. Build logs: https://huggingface.co/spaces/$SPACE?logs=build"
