param(
    [int]$Port = 8765,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Virtual environment not found. Run: python -m venv .venv; .venv\Scripts\pip install -e `".[dev]`""
}

$ArgsList = @("-m", "ferret_rag", "--port", "$Port")
if ($NoBrowser) {
    $ArgsList += "--no-browser"
}

& $Python @ArgsList
