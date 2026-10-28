#!/usr/bin/env bash
# Fetch robot models from MuJoCo Menagerie (Apache-2.0) into third_party/menagerie.
# Pinned to a commit so scenes are reproducible. Used by local setup and the Dockerfile.
set -euo pipefail
COMMIT=f3b6013629c7b94f8ee93fd977b161fa8350f9bc
ROBOTS=(franka_emika_panda)
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/menagerie"

if [ -f "$DEST/.commit" ] && [ "$(cat "$DEST/.commit")" = "$COMMIT" ]; then
  echo "Menagerie already at $COMMIT"; exit 0
fi
rm -rf "$DEST"
git clone -q --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git "$DEST"
git -C "$DEST" sparse-checkout set "${ROBOTS[@]}"
git -C "$DEST" -c advice.detachedHead=false checkout -q "$COMMIT"
rm -rf "$DEST/.git"
echo "$COMMIT" > "$DEST/.commit"
echo "Fetched ${ROBOTS[*]} @ $COMMIT into $DEST"
