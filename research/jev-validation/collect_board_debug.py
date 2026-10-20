from pathlib import Path
import sys,subprocess,json,datetime,time
sys.path.insert(0,'/mnt/f/Files/OPi-Lab/scripts')
import board_access
root=Path(__file__).resolve().parent
out=root/'results'/'board-debug-20261020'
out.mkdir(exist_ok=True,parents=True)
checks={
'identity':"""date -Is; uname -a; cat /proc/sys/kernel/random/boot_id; uptime; tr '\\0' '\\n' < /proc/device-tree/model; cat /etc/os-release | head -6; lscpu | head -25; free -m""",
'network':"""ip -br addr; ip route; ip neigh; iw dev 2>/dev/null; iw dev wlan0 link 2>/dev/null; iw dev wlan0 get power_save 2>/dev/null; systemctl is-active ssh NetworkManager; nmcli -t -f DEVICE,TYPE,STATE,CONNECTION dev status 2>/dev/null; ping -n -c 6 -i 0.3 -W 2 192.168.10.1; ping -n -c 6 -i 0.3 -W 2 192.168.10.13""",
'storage':"""findmnt -no SOURCE,FSTYPE,OPTIONS /; df -h /; df -i /; lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS; dmesg -T | grep -Ei 'mmc|ext4|I/O error|read.only|buffer I/O|corrupt|timeout|reset|voltage|brownout' | tail -100; journalctl --list-boots --no-pager | tail -6""",
'resources':"""free -m; vmstat 1 4; for z in /sys/class/thermal/thermal_zone*; do printf '%s ' "$z"; cat "$z/type" "$z/temp"; done; for c in /sys/devices/system/cpu/cpufreq/policy*; do printf '%s ' "$c"; cat "$c/scaling_governor" "$c/scaling_cur_freq"; done; ps -eo pid,ppid,stat,comm,pcpu,pmem,rss --sort=-pcpu | head -20; cat /proc/pressure/cpu /proc/pressure/memory /proc/pressure/io""",
'services':"""systemctl --failed --no-pager; ss -ltnp; ps -eo pid,args | grep -Ei 'llama|vlm|npu|agent|audio|asr' | grep -v grep | head -18; ls -l /dev/*npu* /dev/galcore /dev/vip* 2>/dev/null; lsmod | grep -Ei 'npu|vip|galcore'; journalctl -b -p err --no-pager -n 35""",
'wifi-log':"""journalctl -u NetworkManager --since '2 hours ago' --no-pager -n 65; journalctl -u ssh --since '30 minutes ago' --no-pager -n 25""",
}
records=[]
for name,cmd in checks.items():
 start=time.monotonic()
 try:
  r=board_access.ssh(cmd,capture_output=True,text=True,timeout=35)
  row=dict(name=name,rc=r.returncode,elapsed_s=time.monotonic()-start,stdout=r.stdout,stderr=r.stderr)
 except Exception as e:row=dict(name=name,error=repr(e),elapsed_s=time.monotonic()-start)
 records.append(row)
 (out/(name+'.txt')).write_text(row.get('stdout','')+'\nSTDERR\n'+row.get('stderr',row.get('error','')))
 print(name,row.get('rc'),round(row['elapsed_s'],2),flush=True)
(out/'checks.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':board_access.HOST,'mode':'read-only diagnostics','checks':records},ensure_ascii=False,indent=2))

