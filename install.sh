#!/usr/bin/env bash
# Jetson ANPR Studio — one-shot installer.
#
# Run this on a fresh NVIDIA Jetson with JetPack 6.x and you'll have
# Sparkler running at http://<jetson-ip>:8080 in about 5 minutes.
#
#   curl -fsSL https://raw.githubusercontent.com/Arijit1080/jetson-anpr-studio/main/install.sh | bash
#
# What it does:
#   1. Sanity-checks that this is a Jetson on a supported JetPack
#   2. Installs docker + nvidia-container-toolkit if missing
#   3. Adds the current user to the docker group (no more `sudo docker`)
#   4. Downloads docker-compose.yml
#   5. Pulls the latest jetson-anpr-studio image and starts it

set -euo pipefail

GH_USER="Arijit1080"
REPO="jetson-anpr-studio"
COMPOSE_URL="https://raw.githubusercontent.com/${GH_USER}/${REPO}/main/docker-compose.yml"
INSTALL_DIR="${HOME}/${REPO}"

log()   { printf "\n\033[1;34m[install]\033[0m %s\n" "$*"; }
warn()  { printf "\n\033[1;33m[install]\033[0m %s\n" "$*"; }
fail()  { printf "\n\033[1;31m[install]\033[0m %s\n" "$*"; exit 1; }

# ----------------------------------------------------------------- 1. sanity
log "Step 1/5 — checking host"
[ "$(uname -m)" = "aarch64" ] || fail "This installer only supports aarch64 (Jetson)."
[ -f /etc/nv_tegra_release ]  || fail "/etc/nv_tegra_release not found — is this a Jetson?"
grep -q "R36" /etc/nv_tegra_release \
    || warn "L4T release tag is not R36.x. The image is built for JetPack 6.x; YMMV."

# ---------------------------------------------------------------- 2. docker
log "Step 2/5 — installing Docker (if missing)"
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

# ----------------------------------------------------------------- 3. group
log "Step 3/5 — ensuring user is in docker group"
if ! groups | grep -q docker; then
    sudo usermod -aG docker "$USER"
    warn "Added '$USER' to the docker group.  You'll need to log out and back in"
    warn "(or run 'newgrp docker') before docker commands work without sudo."
fi

# --------------------------------------------------------------- 4. compose
log "Step 4/5 — fetching docker-compose.yml"
mkdir -p "${INSTALL_DIR}"
curl -fsSL "${COMPOSE_URL}" -o "${INSTALL_DIR}/docker-compose.yml"

# -------------------------------------------------------------------- 5. run
log "Step 5/5 — pulling image and starting Sparkler"
cd "${INSTALL_DIR}"
# Prefer 'docker compose' (v2 plugin); fall back to legacy 'docker-compose'.
if docker compose version >/dev/null 2>&1; then
    DC=(docker compose)
else
    DC=(docker-compose)
fi
sudo -n "${DC[@]}" pull   2>/dev/null || "${DC[@]}" pull
sudo -n "${DC[@]}" up -d  2>/dev/null || "${DC[@]}" up -d

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

==============================================================
EOF
