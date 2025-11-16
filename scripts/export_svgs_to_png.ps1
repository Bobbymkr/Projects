Param(
  [int]$Width = 1920,
  [int]$Height = 1080,
  [string]$InputDir = "docs\diagrams",
  [string]$OutputDir = "docs\diagrams\png"
)

$paths = @(
  "$InputDir\architecture_overview.svg",
  "$InputDir\system_flowchart.svg",
  "$InputDir\roles_responsibilities.svg"
)

$edgeCandidates = @(
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe"
)
$edgePath = $edgeCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $edgePath) { Write-Error "Microsoft Edge not found"; exit 1 }

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$absOutputDir = (Resolve-Path -Path $OutputDir).Path

foreach ($svg in $paths) {
  if (-not (Test-Path $svg)) { Write-Warning "Missing: $svg"; continue }
  $name = [System.IO.Path]::GetFileNameWithoutExtension($svg)
  $outFileName = $name + "_${Width}x${Height}.png"
  $out = Join-Path -Path $absOutputDir -ChildPath $outFileName
  $absSvg = (Resolve-Path -Path $svg).Path
  $uri = 'file:///' + $absSvg.Replace('\','/')
  & "$edgePath" --headless --disable-gpu --window-size="$Width,$Height" --screenshot="$out" "$uri"
  if (Test-Path $out) { Write-Host "Exported: $out" } else { Write-Warning "Failed to export: $svg" }
}
Write-Host "PNG exports written to $OutputDir"