#!/usr/bin/env bash
#
# One command to bring up the whole stack.
#
#   ./run.sh                 bring everything up
#   ./run.sh --search        ...and start hunting for a hand once nodes are live
#   ./run.sh gain:=1.5       any extra args pass straight through to ros2 launch
#
# Checks the Pi is actually serving both ports BEFORE launching, because the
# failure modes otherwise look like ROS bugs: an unreachable arm shows up as
# "Waiting for /arm/jog_enable", and a dead camera as a servo node that never
# sees a frame.
#
# Environment:
#   MYCOBOT_IP        Pi address           (default 192.168.0.15)
#   MYCOBOT_PI_USER   ssh user on the Pi   (default er)
# Both match what scripts/redeploy_pi.sh already uses.

set -euo pipefail

IP="${MYCOBOT_IP:-192.168.0.15}"
PI_USER="${MYCOBOT_PI_USER:-er}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AUTO_SEARCH=0
LAUNCH_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --search) AUTO_SEARCH=1 ;;
        -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
        *) LAUNCH_ARGS+=("$arg") ;;
    esac
done

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- ROS environment -------------------------------------------------------
[ -f /opt/ros/humble/setup.bash ] || die "ROS 2 Humble not found at /opt/ros/humble"
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash

if [ ! -f "$REPO_DIR/install/setup.bash" ]; then
    say "No build found, running colcon build (first run only)..."
    ( cd "$REPO_DIR" && colcon build --symlink-install )
fi
# shellcheck disable=SC1091
source "$REPO_DIR/install/setup.bash"

# ---- Pi reachable ----------------------------------------------------------
say "Checking Pi at $IP ..."
ping -c1 -W2 "$IP" >/dev/null 2>&1 \
    || die "Cannot ping $IP. Is the Pi powered and on the network? Set MYCOBOT_IP if the address changed."

# ---- Pi services -----------------------------------------------------------
# Best effort: if ssh keys are not set up we carry on and let the port checks
# below decide. A missing ssh login is not itself a reason to refuse to launch.
if ssh -o BatchMode=yes -o ConnectTimeout=5 "$PI_USER@$IP" true 2>/dev/null; then
    for svc in mycobot_server mjpg_streamer; do
        if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$PI_USER@$IP" \
                "systemctl is-active --quiet $svc" 2>/dev/null; then
            say "Starting $svc on the Pi..."
            ssh -o BatchMode=yes "$PI_USER@$IP" "sudo systemctl start $svc" \
                || warn "Could not start $svc (needs passwordless sudo). Continuing."
        fi
    done
else
    warn "No passwordless ssh to $PI_USER@$IP; skipping service check."
fi

# ---- Wait for port 8080 (camera) -------------------------------------------
# Stateless HTTP, safe to poll as often as we like.
say "Waiting for camera on $IP:8080 ..."
for i in $(seq 1 30); do
    code="$(curl -s -m 3 -o /dev/null -w '%{http_code}' \
        "http://$IP:8080/?action=snapshot" 2>/dev/null || echo 000)"
    [ "$code" = "200" ] && { say "Camera OK"; break; }
    [ "$i" = "30" ] && die "Camera never came up on $IP:8080. Check: ssh $PI_USER@$IP 'systemctl status mjpg_streamer'"
    sleep 1
done

# ---- Wait for port 9000 (arm) ----------------------------------------------
# server.py is listen(1) -- SINGLE CLIENT. This probe connects and closes
# immediately; anything that holds the socket open locks the driver out for as
# long as it lives. Do not replace this with `nc` without -z, and never leave a
# stray measure_arm.py running against this port.
say "Waiting for arm TCP on $IP:9000 ..."
for i in $(seq 1 30); do
    if python3 -c "
import socket, sys
try:
    socket.create_connection(('$IP', 9000), timeout=2).close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        say "Arm port OK"
        break
    fi
    [ "$i" = "30" ] && die "Arm never came up on $IP:9000. Check: ssh $PI_USER@$IP 'systemctl status mycobot_server'"
    sleep 1
done

# Give server.py a moment to finish closing the probe connection before the
# driver claims the single client slot.
sleep 1

# ---- Optional auto-search --------------------------------------------------
if [ "$AUTO_SEARCH" = "1" ]; then
    (
        for _ in $(seq 1 60); do
            if ros2 service type /servo/search >/dev/null 2>&1; then
                sleep 2   # let the driver finish arming the jog gate
                say "Triggering /servo/search"
                ros2 service call /servo/search std_srvs/srv/Trigger >/dev/null \
                    || warn "/servo/search call failed"
                exit 0
            fi
            sleep 1
        done
        warn "/servo/search never appeared; start it manually."
    ) &
fi

# ---- Go --------------------------------------------------------------------
say "Launching (MYCOBOT_IP=$IP)"
[ "$AUTO_SEARCH" = "1" ] || \
    say "In another terminal: python3 scripts/control_panel.py  (menu: search, home, stop, status)"
export MYCOBOT_IP="$IP"
exec ros2 launch mycobot_bringup servo_demo.launch.py "${LAUNCH_ARGS[@]}"
