@echo off
setlocal
set "PROJECT_ROOT=%~dp0.."
set "PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Virtual environment not found.
  echo Run: python -m venv .venv
  echo Then: .venv\Scripts\pip install -e ".[dev]"
  exit /b 1
)

"%PYTHON%" -c "import PyInstaller" >nul 2>nul
if errorlevel 1 (
  echo PyInstaller is not installed.
  echo Install it with: .venv\Scripts\pip install pyinstaller
  exit /b 1
)

"%PYTHON%" -m PyInstaller --name FerretRAG --noconfirm --clean --add-data "ferret_rag\ui;ferret_rag\ui" --add-data "icons;icons" --add-data ".venv\Lib\site-packages\llama_cpp\lib;llama_cpp\lib" --collect-all chromadb "ferret_rag\__main__.py"
