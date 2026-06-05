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
ALPR_SRC=${ALPR_SRC:-/opt/alpr-src}
RUNS_DIR=${RUNS_DIR:-/app/sparklers_anpr/runs}

# ----- 0. refresh ALPR Python code from the pristine in-image copy -----
# /app/alpr is masked by the `sparkler_alpr` volume so an image upgrade
# alone won't deliver new .py code to users who already populated the
# volume (the volume's old files win).  We keep a pristine copy at
# ${ALPR_SRC} (set by the Dockerfile) and overlay only the *.py files
# onto the volume on every start.  Model files (.pt/.engine) are NOT
# touched, so TRT engine compilation still persists across upgrades.
if [ -d "${ALPR_SRC}" ]; then
    log "refreshing ALPR .py code from ${ALPR_SRC} → ${ALPR_DIR}"
    mkdir -p "${ALPR_DIR}"
    # `find -print0 | xargs cp` is faster + safer than `cp ${SRC}/*.py`
    # which globs at host shell level and can hit ARG_MAX on big dirs.
    find "${ALPR_SRC}" -maxdepth 1 -name '*.py' -print0 | \
        xargs -0 -I{} cp -f {} "${ALPR_DIR}/"
fi

# ----- 1. ensure persistent dirs exist on the mounted volume -----
mkdir -p "${RUNS_DIR}/crops" "${RUNS_DIR}/uploads" "${RUNS_DIR}/frames"

# ----- 1b. wire fast-plate-ocr's hardcoded ~/.cache path into the volume -----
# The fast_plate_ocr library hardcodes its cache at ~/.cache/fast-plate-ocr/
# and ignores any env override.  We symlink that into /opt/models/ which IS
# volume-mounted, so the ~5 MB ONNX persists across container rebuilds AND
# matches the path the library actually writes to.
mkdir -p /root/.cache /opt/models/fast_plate_ocr
if [ ! -L /root/.cache/fast-plate-ocr ]; then
    if [ -d /root/.cache/fast-plate-ocr ]; then
        # Earlier broken run left a real dir.  Migrate its contents into the
        # volume path so the cache survives.
        cp -an /root/.cache/fast-plate-ocr/. /opt/models/fast_plate_ocr/ 2>/dev/null || true
        rm -rf /root/.cache/fast-plate-ocr
    fi
    ln -snf /opt/models/fast_plate_ocr /root/.cache/fast-plate-ocr
fi

# ----- 2. regenerate TRT engines if missing or out-of-date -----
if [ -d "${ALPR_DIR}" ] && ls "${ALPR_DIR}"/*.pt >/dev/null 2>&1; then
    log "checking TensorRT engines in ${ALPR_DIR}"
    python3 /app/regen_engines.py "${ALPR_DIR}" || log "WARN: engine regen failed; will fall back to .pt at runtime"
else
    log "no .pt files under ${ALPR_DIR}; skipping engine regen"
fi

# ----- 3. hand off to the CMD -----
log "starting: $*"
cd /app/sparklers_anpr
exec "$@"
