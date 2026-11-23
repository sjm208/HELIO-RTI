@echo off
rem Run Helio straight from this folder on Windows: double-click it.
rem It uses the folder's own Python setup (.venv, made as the README says),
rem else the Python launcher (py -3.12).
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m helio_qt %*
) else (
    py -3.12 -m helio_qt %*
)
if errorlevel 1 pause
