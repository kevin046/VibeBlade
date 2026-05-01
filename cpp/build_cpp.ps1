# Build VibeBlade native C++ backend on Windows (PowerShell)
# Usage: powershell -ExecutionPolicy Bypass -File cpp\build_cpp.ps1
# Requires: Visual Studio Build Tools (or VS with C++ workload), pybind11

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir  = Join-Path $ScriptDir "build"

# Ensure build dir exists
if (!(Test-Path $BuildDir)) { New-Item -ItemType Directory -Path $BuildDir | Out-Null }

# Find Python
$Python = (Get-Command python).Path
Write-Host "  Python: $Python"

# Find cmake (pip-installed cmake puts it in Scripts/)
$Cmake = $null
$CmakeCandidates = @(
    (Join-Path (Split-Path -Parent $Python) "Scripts\cmake.exe"),
    (Join-Path (Split-Path -Parent $Python) "cmake.exe"),
    (Get-Command cmake -ErrorAction SilentlyContinue).Path
)
foreach ($c in $CmakeCandidates) {
    if ($c -and (Test-Path $c)) {
        $Cmake = $c
        break
    }
}
if (!$Cmake) {
    Write-Host "  cmake not found. Run: pip install cmake" -ForegroundColor Red
    exit 1
}
Write-Host "  CMake: $Cmake"

# Configure
Push-Location $BuildDir
Write-Host "  Configuring..."
& $Cmake .. -DCMAKE_BUILD_TYPE=Release "-DPython3_EXECUTABLE=$Python" -DPYBIND11_FINDPYTHON=ON 2>&1 | ForEach-Object { $_ }
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "  Configure failed" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "  Building..."
& $Cmake --build . --config Release 2>&1 | ForEach-Object { $_ }
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "  Build failed" -ForegroundColor Red
    exit 1
}

# Find .pyd
$PydFile = Get-ChildItem -Path $BuildDir -Recurse -Filter "_vibeblade_native*.pyd" | Select-Object -First 1
if (!$PydFile) {
    Pop-Location
    Write-Host "  No .pyd found - build may have failed" -ForegroundColor Red
    exit 1
}

# Copy to package
$Dest = Join-Path $ScriptDir "..\vibeblade\$($PydFile.Name)"
Copy-Item $PydFile.FullName -Destination $Dest -Force
Write-Host "  Copied $($PydFile.Name) -> vibeblade\$($PydFile.Name)" -ForegroundColor Green

Pop-Location

# Verify
Write-Host "  Verifying import..."
$RootDir = Split-Path -Parent $ScriptDir
Push-Location $RootDir
python -c "import vibeblade._vibeblade_native as nat; print(f'  SIMD: {nat.SIMD_BACKEND}')"
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Native backend ready!" -ForegroundColor Green
} else {
    Write-Host "  Import failed" -ForegroundColor Red
}
Pop-Location
