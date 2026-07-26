#!/usr/bin/env bash
set -euo pipefail

PI_USER="${PI_USER:-er}"
PI_IP="${PI_IP:-192.168.1.169}"

echo "Syncing pi/ folder to ${PI_USER}@${PI_IP}:~/JON/mycobot_project/pi/"
rsync -avz --progress "$(dirname "$0")/../pi/" "${PI_USER}@${PI_IP}:~/JON/mycobot_project/pi/"
echo "Done. SSH in and run: bash ~/JON/mycobot_project/pi/setup_pi.sh"
