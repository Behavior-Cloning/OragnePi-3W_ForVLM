#!/usr/bin/env bash
# Bounded boot diagnostics, no Wi-Fi secrets; persist outside the vendor RAM log.
set -u
log_file=/var/lib/opi-network/boot-network.log
install -d -m 700 /var/lib/opi-network
if [[ -f "$log_file" ]]; then mv -f "$log_file" "$log_file.previous"; fi
exec >"$log_file" 2>&1
chmod 600 "$log_file"
for attempt in $(seq 1 12); do
    date --iso-8601=seconds
    nmcli -f GENERAL.STATE,GENERAL.CONNECTION,GENERAL.HWADDR,IP4.ADDRESS device show wlan0
    nmcli -f IN-USE,SSID,FREQ,CHAN,SIGNAL,SECURITY device wifi list ifname wlan0
    ip -4 route
    if nmcli -g GENERAL.CONNECTION device show wlan0 | grep -Fxq professional-EXT5G &&
       ip -4 addr show wlan0 | grep -q 'inet '; then
        echo 'TARGET_WIFI_CONNECTED'
        break
    fi
    sleep 20
done
journalctl -b -u NetworkManager --no-pager -n 160
sync "$log_file"
