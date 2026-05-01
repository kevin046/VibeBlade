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

# Configure using python -m cmake (pip cmake may not put cmake.exe on PATH)
Push-Location $BuildDir
Write-Host "  Configuring..."
python -m cmake .. -DCMAKE_BUILD_TYPE=Release "-DPython3_EXECUTABLE=$Python" -DPYBIND11_FINDPYTHON=ON 2>&1 | ForEach-Object { $_ }
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Host "  Configure failed" -ForegroundColor Red
    exit 1
}

# Build
Write-Host "  Building..."
python -m cmake --build . --config Release 2>&1 | ForEach-Object { $_ }
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
    Write-Host "  Import failed - checking error..." -ForegroundColor Yellow
    python -c "import traceback; traceback.print_exc()" 2>&1 | ForEach-Object { Write-Host "    $_" }
    python -c "try:`n import vibeblade._vibeblade_native as nat`n print('OK:', dir(nat))`nexcept Exception as e:`n print('ERROR:', type(e).__name__, e)" 2>&1 | ForEach-Object { Write-Host "    $_" }
}
Pop-Location
