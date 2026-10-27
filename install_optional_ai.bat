@echo off
setlocal EnableExtensions
chcp 65001 >nul
title Dataset Curator - Install optional AI dependencies

REM Keep this path synchronized with start_dataset_curator.bat.
set "PYTHON=C:\Program Files\Python\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo ============================================================
echo Dataset Curator - установка зависимостей BLIP / CLIP
echo ============================================================
echo Используемый интерпретатор:
"%PYTHON%" --version
if errorlevel 1 (
    echo [ERROR] Python не найден. Проверьте путь в install_optional_ai.bat.
    pause
    exit /b 1
)

echo.
echo Устанавливаются только transformers и accelerate.
echo PyTorch и CUDA этой командой не переустанавливаются.
"%PYTHON%" -m pip install "transformers>=4.45,<5.0" "accelerate>=0.30"
if errorlevel 1 (
    echo.
    echo [ERROR] Установка завершилась с ошибкой. Проверьте интернет, pip и Python.
    pause
    exit /b 1
)

echo.
"%PYTHON%" -c "import transformers; print('transformers', transformers.__version__)"
if errorlevel 1 (
    echo [ERROR] Пакет не импортируется из выбранного Python.
    pause
    exit /b 1
)
echo.
echo [OK] AI-зависимости установлены. Перезапустите Dataset Curator.
pause
endlocal
