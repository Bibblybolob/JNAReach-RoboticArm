#!/usr/bin/env bash
# Pre-configure WiFi on a Jetson's rootfs so it joins a network on first boot.
#
#     sudo ./scripts/seed_jetson_wifi.sh /media/you/APP "MySSID"
#
# JetPack uses NetworkManager, so there is no Raspberry-Pi-style file you can
# drop on the boot partition -- the config lives on the ROOT filesystem, under
# /etc/NetworkManager/system-connections. Put the Jetson's microSD (or its
# NVMe, via an adapter) into this machine, find the big Linux partition, and
# point this at where it mounted.
#
# WHY A SCRIPT AND NOT A SNIPPET
#
# NetworkManager silently ignores a connection file that is group- or
# world-readable. No error, no log line, no network -- it simply does not
# appear. That, a missing uuid, and the wrong filename extension are three
# ways to end up with a board that boots to nothing and no clue why.
#
# THE PASSPHRASE IS NEVER AN ARGUMENT. It is read from your terminal with
# echo off, so it stays out of your shell history and out of `ps`, where
# anyone on the machine could read it.
#
# AFTER THIS, THE BOARD STILL NEEDS A USER. A freshly flashed JetPack runs an
# oem-config wizard on first boot -- account, locale, licence -- and sits there
# until someone answers it. WiFi alone will not give you SSH if that has never
# been done. This script checks and tells you.
set -euo pipefail

ROOTFS="${1:-}"
SSID="${2:-}"

if [ -z "$ROOTFS" ] || [ -z "$SSID" ]; then
    echo "usage: sudo $0 <path-to-mounted-rootfs> <ssid>" >&2
    echo >&2
    echo "Find the partition first -- it is the large ext4 one:" >&2
    echo "    lsblk -f" >&2
    exit 2
fi

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this with sudo: the file must end up owned by root, and" >&2
    echo "NetworkManager ignores it otherwise." >&2
    exit 1
fi

NM_DIR="$ROOTFS/etc/NetworkManager/system-connections"
if [ ! -d "$ROOTFS/etc/NetworkManager" ]; then
    echo "$ROOTFS does not look like a Linux root filesystem" >&2
    echo "(no etc/NetworkManager). Check what you mounted -- the Jetson's" >&2
    echo "card has several partitions and only the large one is the rootfs." >&2
    exit 1
fi

# A board that has never completed oem-config has no login, so WiFi on its own
# will not get you in. Regular users start at uid 1000.
if [ -f "$ROOTFS/etc/passwd" ]; then
    users=$(awk -F: '$3 >= 1000 && $3 < 65534 { print $1 }' \
            "$ROOTFS/etc/passwd" | tr '\n' ' ')
    if [ -z "$users" ]; then
        echo "WARNING: no user account exists on this image yet."
        echo
        echo "  It will boot into the oem-config setup wizard and wait there,"
        echo "  so WiFi will not give you SSH by itself. Complete first boot"
        echo "  once over a monitor or the serial console, or reflash having"
        echo "  run l4t_create_default_user.sh. This WiFi config will still be"
        echo "  waiting when you do."
        echo
        printf "Carry on anyway? [y/N] "
        read -r reply
        case "$reply" in [yY]*) ;; *) echo "Stopped."; exit 1 ;; esac
    else
        echo "Existing account(s) on this image: $users"
    fi
fi

# Read the passphrase off the terminal rather than argv, and confirm it --
# there is no way to notice a typo later except a board that never appears.
printf "Passphrase for %s (not echoed): " "$SSID"
read -rs psk
echo
printf "Again: "
read -rs psk2
echo
if [ "$psk" != "$psk2" ]; then
    echo "They do not match." >&2
    exit 1
fi
if [ ${#psk} -lt 8 ] || [ ${#psk} -gt 63 ]; then
    echo "WPA passphrases are 8 to 63 characters; that one is ${#psk}." >&2
    exit 1
fi

uuid=$(cat /proc/sys/kernel/random/uuid)
# Filename does not have to match the SSID, but it must end .nmconnection.
safe=$(printf '%s' "$SSID" | tr -c 'A-Za-z0-9._-' '_')
target="$NM_DIR/${safe}.nmconnection"

mkdir -p "$NM_DIR"
umask 077
cat > "$target" <<EOF
[connection]
id=$SSID
uuid=$uuid
type=wifi
autoconnect=true
autoconnect-priority=10

[wifi]
mode=infrastructure
ssid=$SSID

[wifi-security]
key-mgmt=wpa-psk
psk=$psk

[ipv4]
method=auto

[ipv6]
method=auto
addr-gen-mode=default
EOF

chown 0:0 "$target"
chmod 600 "$target"
unset psk psk2

echo
echo "Wrote $target"
ls -l "$target"
echo
echo "  mode 600 and root-owned, which is the part NetworkManager is fussy"
echo "  about -- anything more permissive and it ignores the file in silence."
echo
echo "Unmount cleanly before pulling the card, or the write may not have"
echo "reached it:"
echo
echo "    sudo umount $ROOTFS"
echo
echo "On the Jetson afterwards, to confirm it took:"
echo
echo "    nmcli connection show --active"
