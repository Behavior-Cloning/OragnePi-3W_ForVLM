$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$img = 'F:\Files\OPi-Lab\backups\opi-tf-20261017-original.img'
$base = 'F:\Files\OPi-Lab\repos\OragnePi-3W_ForVLM\research\vlm-molmo-paligemma-20261023\maintenance'
$expected = '93c6a03aa6c7b527a148c2ed1e68d77bfa2de3820df302e6c2c936a66ebb4087'
$record = [ordered]@{ action='NTFS transparent LZX compression; preserve full TF image'; start=(Get-Date).ToString('o'); free_before=(Get-Volume -DriveLetter F).SizeRemaining; image_bytes=(Get-Item -LiteralPath $img).Length; status='compressing' }
$record | ConvertTo-Json | Set-Content -Encoding UTF8 "$base\tf-compression.json"
& compact.exe /C /I /EXE:LZX $img *> "$base\tf-compression.log"
$record.compact_exit = $LASTEXITCODE
$record.status='verifying_sha256'
$record | ConvertTo-Json | Set-Content -Encoding UTF8 "$base\tf-compression.json"
$actual = (Get-FileHash -LiteralPath $img -Algorithm SHA256).Hash.ToLowerInvariant()
$record.sha256 = $actual
$record.matches_original_sha256 = ($actual -eq $expected)
$record.free_after = (Get-Volume -DriveLetter F).SizeRemaining
$record.finish = (Get-Date).ToString('o')
$record.status = if (($record.compact_exit -eq 0) -and ($actual -eq $expected)) {'compressed_verified'} else {'review_required'}
$record | ConvertTo-Json | Set-Content -Encoding UTF8 "$base\tf-compression.json"
if ($actual -ne $expected) { throw 'Backup hash mismatch; preserve file and investigate' }
