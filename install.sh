#!/usr/bin/env bash
# Jetson ANPR Studio — one-shot installer.
#
# Run this on a fresh NVIDIA Jetson with JetPack 6.x and you'll have
# Sparkler running at http://<jetson-ip>:8080 in about 5 minutes.
#
#   curl -fsSL https://raw.githubusercontent.com/Arijit1080/jetson-anpr-studio/main/install.sh | bash
#
# Or with a GitHub token if the image is still private (until you flip the
# GHCR package to public):
#
#   curl -fsSL .../install.sh | GHCR_TOKEN=ghp_xxx GHCR_USER=Arijit1080 bash
#
# What it does:
#   1. Sanity-checks Jetson hardware + JetPack version
#   2. Installs docker + nvidia-container-toolkit if missing
#   3. Adds the current user to the docker group (or uses sg docker for this session)
#   4. Stops any bare-metal sparkler systemd service (port 8080 conflict)
#   5. Authenticates to GHCR if GHCR_TOKEN env var is set
#   6. Downloads docker-compose.yml
#   7. Pulls the latest jetson-anpr-studio image and starts it
#
# Environment variables (all optional):
#   GHCR_TOKEN  — GitHub Personal Access Token with read:packages scope.
#                 Only needed while the GHCR image is private.
#   GHCR_USER   — your GitHub username (defaults to Arijit1080).
#   INSTALL_DIR — where to put the compose file (defaults to ~/jetson-anpr-studio).

set -euo pipefail

GH_USER="Arijit1080"
REPO="jetson-anpr-studio"
COMPOSE_URL="https://raw.githubusercontent.com/${GH_USER}/${REPO}/main/docker-compose.yml"
IMAGE="ghcr.io/${GH_USER,,}/${REPO}:latest"
INSTALL_DIR="${INSTALL_DIR:-${HOME}/${REPO}}"
GHCR_TOKEN="${GHCR_TOKEN:-}"
GHCR_USER="${GHCR_USER:-${GH_USER}}"

log()   { printf "\n\033[1;34m[install]\033[0m %s\n" "$*"; }
warn()  { printf "\n\033[1;33m[install]\033[0m %s\n" "$*"; }
fail()  { printf "\n\033[1;31m[install]\033[0m %s\n" "$*"; exit 1; }

# A small wrapper that runs `docker` either directly (if user already in
# docker group) or via `sg docker -c '...'` to side-step the chicken-and-egg
# "added to group but current shell doesn't know yet" problem.
docker_run() {
    if groups | grep -q docker; then
        docker "$@"
    elif command -v sg >/dev/null 2>&1; then
        sg docker -c "docker $*"
    else
        sudo docker "$@"
    fi
}

# ----------------------------------------------------------------- 1. sanity
log "Step 1/7 — checking host"
[ "$(uname -m)" = "aarch64" ] || fail "This installer only supports aarch64 (Jetson)."
[ -f /etc/nv_tegra_release ]  || fail "/etc/nv_tegra_release not found — is this a Jetson?"
grep -q "R36" /etc/nv_tegra_release \
    || warn "L4T release tag is not R36.x. The image is built for JetPack 6.x; YMMV."

FREE_GB=$(df --output=avail / | tail -1 | awk '{print int($1/1024/1024)}')
[ "${FREE_GB}" -ge 8 ] \
    || warn "Only ${FREE_GB} GB free on /. The image is ~6.7 GB; consider freeing space (see README)."

# ---------------------------------------------------------------- 2. docker
log "Step 2/7 — installing Docker (if missing)"
if ! command -v docker >/dev/null 2>&1; then
    curl -fsSL https://get.docker.com | sudo sh
fi

log "         installing nvidia-container-toolkit (if missing)"
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Verify the nvidia runtime is registered with the daemon
if ! sudo docker info 2>/dev/null | grep -qi nvidia; then
    warn "Docker daemon doesn't report the nvidia runtime. Re-configuring..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# ----------------------------------------------------------------- 3. group
log "Step 3/7 — adding user to docker group"
if ! groups | grep -q docker; then
    sudo usermod -aG docker "$USER"
    warn "Added '$USER' to the docker group. This session uses 'sg docker' as a workaround"
    warn "so you don't have to log out. Future sessions: docker commands will Just Work."
fi

# ----------------------------------------------------------------- 4. port
log "Step 4/7 — checking port 8080"
if sudo systemctl is-active --quiet sparkler 2>/dev/null; then
    warn "A bare-metal 'sparkler' systemd service is running on port 8080."
    warn "Stopping it so the container can bind to that port."
    sudo systemctl stop sparkler
    sudo systemctl disable sparkler 2>/dev/null || true
fi
if ss -tln 2>/dev/null | grep -q ":8080 "; then
    fail "Port 8080 is in use by something else. Free it before running install.sh."
fi

# -------------------------------------------------------------- 5. ghcr auth
log "Step 5/7 — registry authentication"
# Probe whether the image is publicly pullable.  HTTP 200 = public; 401 = private.
HTTP=$(curl -s -o /dev/null -w '%{http_code}' \
    "https://ghcr.io/v2/${GHCR_USER,,}/${REPO}/manifests/latest" || true)
if [ "${HTTP}" = "200" ]; then
    log "         image is public — no login needed"
elif [ -n "${GHCR_TOKEN}" ]; then
    log "         logging in to ghcr.io with provided GHCR_TOKEN"
    echo "${GHCR_TOKEN}" | docker_run login ghcr.io -u "${GHCR_USER}" --password-stdin
else
    cat <<EOF

WARNING: The GHCR image still appears to be private (manifest probe returned HTTP ${HTTP}).
To pull it, you need ONE of:

  (a) Flip the package to public — recommended:
      https://github.com/${GH_USER}/${REPO}/pkgs/container/${REPO}
      → Package settings → Danger Zone → Change visibility → Public

  (b) Re-run this installer with a GitHub token:
      curl -fsSL .../install.sh | GHCR_TOKEN=ghp_xxx GHCR_USER=${GH_USER} bash

Proceeding anyway — the pull below will fail if you haven't done (a) or (b).

EOF
fi

# ---------------------------------------------------------------- 6. compose
log "Step 6/7 — fetching docker-compose.yml"
mkdir -p "${INSTALL_DIR}"
curl -fsSL "${COMPOSE_URL}" -o "${INSTALL_DIR}/docker-compose.yml"

# -------------------------------------------------------------------- 7. run
log "Step 7/7 — pulling image and starting Sparkler"
cd "${INSTALL_DIR}"
docker_run compose pull
docker_run compose up -d

IP=$(hostname -I | awk '{print $1}')
cat <<EOF

==============================================================
 Jetson ANPR Studio is starting up.

 Open in your browser:

   http://${IP:-<jetson-ip>}:8080

 First start regenerates TensorRT engines (~2 min).  Watch logs with:

   docker compose -f ${INSTALL_DIR}/docker-compose.yml logs -f

 Stop / start:

   docker compose -f ${INSTALL_DIR}/docker-compose.yml down
   docker compose -f ${INSTALL_DIR}/docker-compose.yml up -d

 Upgrade to a newer image (after a code change is pushed to main):

   docker compose -f ${INSTALL_DIR}/docker-compose.yml pull
   docker compose -f ${INSTALL_DIR}/docker-compose.yml up -d

==============================================================
EOF
