@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Dataset Curator - Install optional translation SDKs

REM Keep this path synchronized with start_dataset_curator.bat.
set "PYTHON=C:\Program Files\Python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo ============================================================
echo Dataset Curator - optional translation SDKs
echo ============================================================
"%PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] Python не найден. Проверьте путь в install_translation_optional.bat.
    pause
    exit /b 1
)
echo.
echo Устанавливаются SDK для Amazon Translate и Alibaba Cloud.
echo Остальным провайдерам достаточно requests из requirements.txt.
"%PYTHON%" -m pip install -r requirements_translation_optional.txt
if errorlevel 1 (
    echo.
    echo [ERROR] Установка SDK завершилась с ошибкой.
    pause
    exit /b 1
)
echo.
echo [OK] Дополнительные SDK установлены.
pause
endlocal
