param(
    [string]$Name = "FerretRAG"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .venv\Scripts\pip install -e `".[dev]`""
}

& $Python -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller is not installed. Install it in the venv before packaging: .venv\Scripts\pip install pyinstaller"
}

$IconPath = Join-Path $ProjectRoot "icons\icon_round.png"
if (Test-Path $IconPath) {
    Write-Host "Using PNG icon as bundled asset. Windows EXE icon conversion is a later packaging polish step."
}

& $Python -m PyInstaller `
    --name $Name `
    --noconfirm `
    --clean `
    --add-data "ferret_rag\ui;ferret_rag\ui" `
    --add-data "icons;icons" `
    --collect-all chromadb `
    "ferret_rag\__main__.py"

Write-Host "Build complete. Check dist\$Name."
