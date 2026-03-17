param(
  [Parameter(Mandatory = $true)]
  [string]$Artifact,

  [Parameter(Mandatory = $true)]
  [string]$Triplet
)

$ErrorActionPreference = "Stop"

Remove-Item package -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path package, package\LICENSES | Out-Null

$exe = "build/molspin.exe"
if (-not (Test-Path $exe)) {
  throw "molspin.exe not found at $exe."
}

Copy-Item $exe package\molspin.exe
Copy-Item README.md package\README.md

$dllSource = Join-Path $env:VCPKG_ROOT "installed/$Triplet/bin"
if (Test-Path $dllSource) {
  Get-ChildItem -Path $dllSource -File -Filter *.dll | ForEach-Object {
    Copy-Item $_.FullName package\
  }
}

$licenseRoots = @(
  Join-Path $env:VCPKG_ROOT "installed/$Triplet/share",
  Join-Path $env:VCPKG_ROOT "packages"
)

$copied = @{}
foreach ($root in $licenseRoots) {
  if (-not (Test-Path $root)) {
    continue
  }

  Get-ChildItem -Path $root -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match '^(LICENSE|LICENCE|COPYING|copyright)(\..*)?$' } |
    ForEach-Object {
      $relativeParent = $_.DirectoryName.Substring($root.Length).TrimStart('\')
      $safeParent = if ($relativeParent) { $relativeParent -replace '[\\/:*?"<>|]', '_' } else { "root" }
      $destName = "$safeParent-$($_.Name)"
      if (-not $copied.ContainsKey($destName)) {
        Copy-Item $_.FullName (Join-Path package\LICENSES $destName) -ErrorAction SilentlyContinue
        $copied[$destName] = $true
      }
    }
}

Compress-Archive -Path package\* -DestinationPath $Artifact -Force
