@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Dataset Curator

REM Keep this path synchronized with install_optional_ai.bat and update.bat.
set "PYTHON=C:\Program Files\Python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo Dataset Curator
echo Python: 
"%PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] Python не найден.
    pause
    exit /b 1
)
echo.
"%PYTHON%" "%~dp0dataset_curator_gui.py"
if errorlevel 1 (
    echo.
    echo [INFO] Приложение завершилось с ошибкой. Проверьте журнал выше.
    pause
)
endlocal
