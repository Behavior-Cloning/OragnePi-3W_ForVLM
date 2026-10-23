# Run only after all experimental jobs have completed and WSL data is synced.
# This operates on OPiUbuntu only; it does not unregister or resize the filesystem.
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$dir = 'F:\Files\OPi-Lab\repos\OragnePi-3W_ForVLM\research\vlm-molmo-paligemma-20261023\maintenance'
$vhd = 'F:\Files\OPi-Lab\wsl\Ubuntu2204\ext4.vhdx'
function Snapshot {
  $disk = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='F:'"
  @{ utc=(Get-Date).ToUniversalTime().ToString('o'); free_F_bytes=[int64]$disk.FreeSpace; vhd_length_bytes=[int64](Get-Item $vhd).Length }
}
$record = @{ distro='OPiUbuntu'; before=(Snapshot); status='starting' }
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$dir\wsl-compact.json"
& wsl.exe --terminate OPiUbuntu 2>&1 | Out-File -Encoding UTF8 "$dir\wsl-terminate.log"
$record.terminate_exit = $LASTEXITCODE
if ($record.terminate_exit -ne 0) { throw 'WSL termination failed; compaction skipped' }
$timer = [Diagnostics.Stopwatch]::StartNew()
& wsl.exe --manage OPiUbuntu --compact 2>&1 | Out-File -Encoding UTF8 "$dir\wsl-compact.log"
$record.compact_exit = $LASTEXITCODE
$timer.Stop()
$record.elapsed_seconds = $timer.Elapsed.TotalSeconds
$record.after = Snapshot
$record.vhd_reduction_bytes = $record.before.vhd_length_bytes - $record.after.vhd_length_bytes
$record.free_F_increase_bytes = $record.after.free_F_bytes - $record.before.free_F_bytes
$record.status = if ($record.compact_exit -eq 0) { 'compact_command_succeeded_restart_verification_pending' } else { 'compact_command_failed' }
$record | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 "$dir\wsl-compact.json"
$record | ConvertTo-Json -Depth 6
