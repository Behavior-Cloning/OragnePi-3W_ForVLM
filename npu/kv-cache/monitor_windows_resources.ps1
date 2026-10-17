param([Parameter(Mandatory=$true)][string]$OutputDirectory,
      [Parameter(Mandatory=$true)][string]$VhdPath,
      [string]$Drive='F', [int]$IntervalSeconds=5)
$ErrorActionPreference='Stop'
$ProgressPreference='SilentlyContinue'
if ($IntervalSeconds -lt 1) { throw 'Interval must be positive' }
New-Item -ItemType Directory -Force $OutputDirectory | Out-Null
while (-not (Test-Path (Join-Path $OutputDirectory 'monitor.stop'))) {
    $os=Get-CimInstance Win32_OperatingSystem
    $vm=Get-Process -Name '*vmmem*' -ErrorAction SilentlyContinue
    $disk=Get-PSDrive $Drive
    $row=[ordered]@{
        utc=[DateTime]::UtcNow.ToString('o')
        total_bytes=$os.TotalVisibleMemorySize*1024
        available_bytes=$os.FreePhysicalMemory*1024
        vmmem_bytes=($vm|Measure-Object WorkingSet64 -Sum).Sum
        f_free_bytes=$disk.Free
        vhd_bytes=(Get-Item $VhdPath).Length
    }
    ($row|ConvertTo-Json -Compress)|Add-Content -Encoding UTF8 (Join-Path $OutputDirectory 'windows-resources.jsonl')
    Start-Sleep -Seconds $IntervalSeconds
}
