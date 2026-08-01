#!/usr/bin/env bash
# Create the first user on a Jetson rootfs, so it boots past the setup wizard.
#
#     sudo ./scripts/seed_jetson_user.sh /mnt/jetson jonathan
#
# A freshly flashed JetPack boots into nv-oem-config.target and runs a wizard
# that wants a display. If you have no monitor, that is a wall: the board comes
# up, answers ping, serves DHCP over USB -- and refuses SSH forever, because
# nv-oem-config.target declares Conflicts=multi-user.target and ssh.service is
# wanted by multi-user.target. It can never start.
#
# The USB serial console is no help either, for the same reason. On R36,
# serial-getty@ttyGS0 IS enabled, but getty.target is reached through
# multi-user.target, so nothing is listening on /dev/ttyACM0 from the host's
# side. A blank screen with no response to Enter is the expected symptom, not
# a wiring fault.
#
# So this does what the wizard would have done: creates the account, then
# repoints default.target at multi-user.target. Everything else -- ssh,
# the USB console getty, NetworkManager -- is already enabled on the image and
# starts on its own once that target is reachable.
#
# Verified against L4T R36.3.0 (JetPack 6.0), Ubuntu 22.04.4. Re-check
# /etc/systemd/system/default.target on other releases before trusting it.
#
# Pair with seed_jetson_wifi.sh and the board comes up on the network with a
# working login, having never needed a screen.
set -euo pipefail

ROOTFS="${1:-}"
USERNAME="${2:-}"
TARGET="${3:-multi-user.target}"

if [ -z "$ROOTFS" ] || [ -z "$USERNAME" ]; then
    echo "usage: sudo $0 <mounted-rootfs> <username> [default-target]" >&2
    echo >&2
    echo "  default-target is multi-user.target (headless, the default here)" >&2
    echo "  or graphical.target if you will attach a monitor." >&2
    exit 2
fi
if [ "$(id -u)" -ne 0 ]; then
    echo "Run with sudo -- this writes /etc/passwd and /etc/shadow." >&2
    exit 1
fi
if [ ! -f "$ROOTFS/etc/passwd" ] || [ ! -d "$ROOTFS/etc/systemd/system" ]; then
    echo "$ROOTFS does not look like a Linux root filesystem." >&2
    echo "Mount the large ext4 partition of the Jetson's card; lsblk -f." >&2
    exit 1
fi
if ! grep -qE "^${USERNAME}:" "$ROOTFS/etc/passwd"; then :; else
    echo "User '$USERNAME' already exists on this image. Nothing to do." >&2
    exit 1
fi
existing=$(awk -F: '$3>=1000 && $3<65534 {print $1}' "$ROOTFS/etc/passwd")
if [ -n "$existing" ]; then
    echo "This image already has a regular user: $existing" >&2
    echo "Setup has been completed; you should be able to log in already." >&2
    exit 1
fi

UID_NEW=1000
GID_NEW=1000
HOME_DIR="/home/$USERNAME"
# Groups the Ubuntu/L4T desktop user normally gets. dialout is the one this
# project cares about -- without it /dev/ttyTHS1 is unopenable and every
# arm script reports a permission error that reads like a wiring fault.
GROUPS_WANTED="adm dialout cdrom sudo audio dip video plugdev i2c gpio render netdev lpadmin sambashare"

echo "Creating '$USERNAME' (uid $UID_NEW) on $ROOTFS"
echo

printf "Password for %s (not echoed): " "$USERNAME"
read -rs pw; echo
printf "Again: "
read -rs pw2; echo
[ "$pw" = "$pw2" ] || { echo "They do not match." >&2; exit 1; }
[ -n "$pw" ] || { echo "Empty password refused -- sudo would be unusable." >&2; exit 1; }

# Hash via stdin, never argv: anything on a command line is visible in ps to
# every user on the machine for as long as the process lives.
hash=$(printf '%s' "$pw" | openssl passwd -6 -stdin)
unset pw pw2
days=$(( $(date +%s) / 86400 ))

# --- account ------------------------------------------------------------
printf '%s:x:%s:%s:%s,,,:%s:/bin/bash\n' \
    "$USERNAME" "$UID_NEW" "$GID_NEW" "$USERNAME" "$HOME_DIR" \
    >> "$ROOTFS/etc/passwd"
printf '%s:%s:%s:0:99999:7:::\n' "$USERNAME" "$hash" "$days" \
    >> "$ROOTFS/etc/shadow"
printf '%s:x:%s:\n' "$USERNAME" "$GID_NEW" >> "$ROOTFS/etc/group"
printf '%s:!::\n' "$USERNAME" >> "$ROOTFS/etc/gshadow"
unset hash

add_to_group() {
    local grp="$1" f="$ROOTFS/etc/group"
    grep -qE "^${grp}:" "$f" || return 0
    # Append to the member list, handling both empty and populated ones.
    if grep -qE "^${grp}:[^:]*:[^:]*:$" "$f"; then
        sed -i "s/^${grp}:\([^:]*\):\([^:]*\):$/${grp}:\1:\2:${USERNAME}/" "$f"
    elif ! grep -qE "^${grp}:[^:]*:[^:]*:.*\b${USERNAME}\b" "$f"; then
        sed -i "s/^${grp}:\([^:]*\):\([^:]*\):\(.*\)$/${grp}:\1:\2:\3,${USERNAME}/" "$f"
    fi
    echo "    + $grp"
}
echo "  groups:"
for g in $GROUPS_WANTED; do add_to_group "$g"; done

# --- home ---------------------------------------------------------------
mkdir -p "$ROOTFS$HOME_DIR"
if [ -d "$ROOTFS/etc/skel" ]; then
    cp -a "$ROOTFS/etc/skel/." "$ROOTFS$HOME_DIR/" 2>/dev/null || true
fi
chown -R "$UID_NEW:$GID_NEW" "$ROOTFS$HOME_DIR"
chmod 755 "$ROOTFS$HOME_DIR"
echo "  home:   $HOME_DIR (from /etc/skel)"

# --- optional ssh key ---------------------------------------------------
# Offered because a key means the first login cannot be locked out by a
# mistyped password, on a board with no console to fix it from.
key=""
for k in "${SUDO_USER:+/home/$SUDO_USER}"/.ssh/id_*.pub /root/.ssh/id_*.pub; do
    [ -f "$k" ] && { key="$k"; break; }
done
if [ -n "$key" ]; then
    printf "Also install %s for passwordless SSH? [Y/n] " "$(basename "$key")"
    read -r reply
    case "$reply" in
        [nN]*) echo "  skipped" ;;
        *)
            install -d -m 700 -o "$UID_NEW" -g "$GID_NEW" \
                "$ROOTFS$HOME_DIR/.ssh"
            cat "$key" >> "$ROOTFS$HOME_DIR/.ssh/authorized_keys"
            chown "$UID_NEW:$GID_NEW" "$ROOTFS$HOME_DIR/.ssh/authorized_keys"
            chmod 600 "$ROOTFS$HOME_DIR/.ssh/authorized_keys"
            echo "  ssh key installed from $key"
            ;;
    esac
fi

# --- boot past the wizard ----------------------------------------------
# The whole point. default.target ships pointing at nv-oem-config.target,
# which Conflicts=multi-user.target, so ssh.service and getty.target never
# start no matter how healthy the rest of the system is.
old=$(readlink "$ROOTFS/etc/systemd/system/default.target" 2>/dev/null || echo "unset")
ln -sf "/lib/systemd/system/$TARGET" \
    "$ROOTFS/etc/systemd/system/default.target"
echo
echo "  default.target: $old -> $TARGET"

# Un-ban apt sources if this image has them disabled. R36.3.0 does not, but
# nv-oem-config-post.sh does it, so match the wizard's behaviour.
banned=$(find "$ROOTFS/etc/apt" -name '*.banned' 2>/dev/null | wc -l)
if [ "$banned" -gt 0 ]; then
    find "$ROOTFS/etc/apt" -type f -name '*.list.banned' \
        -exec bash -c 'mv "$0" "${0%.banned}"' {} \;
    echo "  apt: re-enabled $banned banned source list(s)"
fi

sync
echo
echo "Done. Unmount before pulling the card:"
echo
echo "    sudo umount $ROOTFS"
echo
echo "Then boot it. ssh.service and serial-getty@ttyGS0 are already enabled"
echo "on this image and will start once $TARGET is reached, so you get both"
echo "the network and the USB console without touching anything else."
echo
echo "    ssh $USERNAME@192.168.55.1        # over the USB-C cable"
