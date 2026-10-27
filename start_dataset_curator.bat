@echo off
setlocal
set "PYTHON=C:\Program Files\Python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
"%PYTHON%" "%~dp0dataset_curator_gui.py"
if errorlevel 1 pause
endlocal
