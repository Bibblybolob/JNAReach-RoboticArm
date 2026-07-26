#!/usr/bin/env bash
# Push the Pi-side scripts and restart both services.
#
# Use this when:
#   - the arm refuses connections ("locked out")
#   - the camera stream will not load
#   - you have changed anything in pi/
#
# WHY THINGS GET STUCK
#
# server.py accepts exactly one client. Until the fix in this repo it had no
# recv timeout, so a client that died without closing -- a killed ROS node, a
# crashed script, a sleeping laptop -- left the server blocked in recv()
# forever. It never returned to accept(), so nothing could reconnect and the
# arm looked dead. The same shape of bug hit the camera: a single-threaded
# HTTP server holds the port while one dead client is still "connected".
#
# Both are fixed in the current pi/ scripts, but the fixes only take effect
# once they are actually ON the Pi -- which is what this script does.
#
# Usage:
#   ./scripts/redeploy_pi.sh                 # uses $MYCOBOT_IP or the default
#   ./scripts/redeploy_pi.sh 192.168.1.46
#   MYCOBOT_PI_USER=pi ./scripts/redeploy_pi.sh

set -euo pipefail

PI_IP="${1:--e}"
PI_USER="${MYCOBOT_PI_USER:-er}"
PI_DIR="${MYCOBOT_PI_DIR:-~/JON/mycobot_project/pi}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Target: ${PI_USER}@${PI_IP}:${PI_DIR}"
echo

if ! ping -c 1 -W 2 "$PI_IP" >/dev/null 2>&1; then
    echo "ERROR: ${PI_IP} is not responding to ping."
    echo
    echo "  - Is the Pi powered on and on the network?"
    echo "  - Are you on the same subnet? A host on 192.168.0.x cannot reach"
    echo "    192.168.1.x without a route between them."
    echo "  - Find the Pi's real address by checking your router, or on the Pi:"
    echo "        hostname -I"
    exit 1
fi
echo "Pi is reachable."

echo "Copying pi/ scripts..."
scp -q "${REPO_DIR}/pi/server.py" "${REPO_DIR}/pi/camera_stream.py" \
    "${PI_USER}@${PI_IP}:${PI_DIR}/"

echo "Installing systemd units..."
scp -q "${REPO_DIR}/pi/mycobot_server.service" "${REPO_DIR}/pi/mjpg_streamer.service" \
    "${PI_USER}@${PI_IP}:/tmp/"
ssh "${PI_USER}@${PI_IP}" \
    'sudo mv /tmp/mycobot_server.service /tmp/mjpg_streamer.service /etc/systemd/system/ && sudo systemctl daemon-reload'

echo "Restarting services..."
# Stop both first: a stale process holding port 9000 or 8080 will make the new
# one fail to bind, which looks identical to the lockout being unfixed.
ssh "${PI_USER}@${PI_IP}" bash -s <<'REMOTE'
set -uo pipefail
sudo systemctl stop mycobot_server.service  2>/dev/null || true
sudo systemctl stop mjpg_streamer.service   2>/dev/null || true

# Kill anything still squatting on the ports. Without this a manually started
# server.py from an earlier debugging session survives the systemctl stop and
# keeps the port, and the restart silently fails to take effect.
sudo fuser -k 9000/tcp 2>/dev/null || true
sudo fuser -k 8080/tcp 2>/dev/null || true
sleep 1

sudo systemctl start mycobot_server.service
sudo systemctl start mjpg_streamer.service
sleep 2

echo
echo "  arm server (9000):  $(systemctl is-active mycobot_server.service)"
echo "  camera     (8080):  $(systemctl is-active mjpg_streamer.service)"
echo
echo "  listening ports:"
ss -tlnp 2>/dev/null | grep -E ':(9000|8080)' || echo "    NONE -- services are not listening, check: journalctl -u mycobot_server -n 30"
REMOTE

echo
echo "If a service is not active, the reason is in its log:"
echo "    ssh ${PI_USER}@${PI_IP} 'journalctl -u mycobot_server -n 30 --no-pager'"
echo
echo "Done. Verify from here:"
echo "    curl -s -m 3 -o /dev/null -w 'camera HTTP %{http_code}\\n' 'http://${PI_IP}:8080/?action=snapshot'"
echo "    python3 ${REPO_DIR}/scripts/measure_arm.py --ip ${PI_IP} --skip-motion"
