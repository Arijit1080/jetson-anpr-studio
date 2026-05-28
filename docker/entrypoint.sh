#!/usr/bin/env bash
# Jetson ANPR Studio — container entrypoint.
#
# Runs on every container start:
#   1. If `.engine` files are missing (or TensorRT version has changed since
#      they were built), regenerate them from the bundled `.pt` weights.
#      This makes the image portable across TRT versions within a JetPack
#      major series.
#   2. Make sure the runtime data directories exist on the mounted volume.
#   3. Exec the CMD (uvicorn by default).
#
# Errors during engine regen are logged but NOT fatal — Sparkler can still
# load `.pt` weights directly, just slower at inference time.

set -euo pipefail

log() { echo "[entrypoint] $*"; }

ALPR_DIR=${ALPR_DIR:-/app/alpr}
RUNS_DIR=${RUNS_DIR:-/app/sparkler/runs}

# ----- 1. ensure persistent dirs exist on the mounted volume -----
mkdir -p "${RUNS_DIR}/crops" "${RUNS_DIR}/uploads" "${RUNS_DIR}/frames"

# ----- 2. regenerate TRT engines if missing or out-of-date -----
if [ -d "${ALPR_DIR}" ] && ls "${ALPR_DIR}"/*.pt >/dev/null 2>&1; then
    log "checking TensorRT engines in ${ALPR_DIR}"
    python3 /app/regen_engines.py "${ALPR_DIR}" || log "WARN: engine regen failed; will fall back to .pt at runtime"
else
    log "no .pt files under ${ALPR_DIR}; skipping engine regen"
fi

# ----- 3. hand off to the CMD -----
log "starting: $*"
cd /app/sparkler
exec "$@"
