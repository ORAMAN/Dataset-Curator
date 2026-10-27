@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title Dataset Curator - Update Repository

REM ============================================================
REM Dataset Curator: update Git clone or ZIP copy without deleting
REM user settings, downloaded models, reports or dataset folders.
REM ZIP updates never delete files that are absent from the new archive.
REM ============================================================

cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
set "REPO_URL=https://github.com/ORAMAN/Dataset-Curator.git"
set "BRANCH=main"
set "ZIP_URL=https://github.com/ORAMAN/Dataset-Curator/archive/refs/heads/main.zip"
set "TMP_DIR=%TEMP%\DatasetCurator_update_%RANDOM%_%RANDOM%"

REM Clear Git path overrides inherited from a parent process.
set "GIT_DIR="
set "GIT_WORK_TREE="
set "GIT_INDEX_FILE="
set "GIT_COMMON_DIR="
set "GIT_OBJECT_DIRECTORY="
set "GIT_ALTERNATE_OBJECT_DIRECTORIES="

echo.
echo ============================================================
echo Dataset Curator - обновление репозитория
echo ============================================================
echo [INFO] Папка: %PROJECT_ROOT%
echo.

if exist ".git" goto :git_update
if exist "dataset_curator_gui.py" if exist "requirements.txt" goto :zip_update
goto :new_install

:git_update
echo [STEP] Обнаружен Git-клон. Проверяю Git for Windows...
call :ensure_git
if errorlevel 1 (
    echo [WARN] Git недоступен. Перехожу к безопасному ZIP-обновлению файлов.
    goto :zip_update
)

git rev-parse --is-inside-work-tree >nul 2>&1
if not errorlevel 1 goto :git_repository_ok
echo [WARN] Git не распознаёт каталог. Пробую настроить safe.directory...
git config --global --add safe.directory "%PROJECT_ROOT%" >nul 2>&1
git rev-parse --is-inside-work-tree >nul 2>&1
if errorlevel 1 (
    echo [WARN] Репозиторий Git всё ещё не распознаётся. Перехожу к ZIP-обновлению.
    goto :zip_update
)
:git_repository_ok

set "CURRENT_BRANCH="
for /f "delims=" %%B in ('git branch --show-current 2^>nul') do set "CURRENT_BRANCH=%%B"
if /I not "!CURRENT_BRANCH!"=="%BRANCH%" (
    echo [ERROR] Текущая ветка: "!CURRENT_BRANCH!", ожидается "%BRANCH%".
    echo Переключитесь на main и повторите обновление.
    goto :error
)

set "ORIGIN="
for /f "delims=" %%R in ('git config --get remote.origin.url 2^>nul') do set "ORIGIN=%%R"
if /I "!ORIGIN!"=="https://github.com/ORAMAN/Dataset-Curator.git" goto :origin_ok
if /I "!ORIGIN!"=="https://github.com/ORAMAN/Dataset-Curator" goto :origin_ok
if /I "!ORIGIN!"=="git@github.com:ORAMAN/Dataset-Curator.git" goto :origin_ok
if /I "!ORIGIN!"=="git@github.com:ORAMAN/Dataset-Curator" goto :origin_ok

echo [ERROR] origin не соответствует ORAMAN/Dataset-Curator:
echo         !ORIGIN!
echo Обновление отменено, чтобы не менять другой репозиторий.
goto :error

:origin_ok
git diff --quiet
if errorlevel 1 goto :local_changes
git diff --cached --quiet
if errorlevel 1 goto :local_changes
goto :fetch_and_update

:local_changes
echo.
echo [WARN] Найдены локальные изменения в отслеживаемых файлах:
git status --short --untracked-files=no
echo.
echo [1] Отмена обновления.
echo [2] Принудительно обновить main и перезаписать локальные правки отслеживаемых файлов.
echo.
:local_changes_prompt
set "LOCAL_CHANGE_CHOICE="
set /p "LOCAL_CHANGE_CHOICE=Ваш выбор [1/2]: "
if "!LOCAL_CHANGE_CHOICE!"=="1" goto :cancelled
if "!LOCAL_CHANGE_CHOICE!"=="2" goto :force_update
echo [WARN] Введите 1 или 2.
goto :local_changes_prompt

:fetch_and_update
echo [STEP] Получение изменений с GitHub...
git fetch --prune origin
if errorlevel 1 (
    echo [ERROR] Не удалось получить изменения с GitHub.
    goto :error
)
echo [STEP] Обновление без сброса локальных файлов (fast-forward)...
git merge --ff-only origin/%BRANCH%
if errorlevel 1 (
    echo [ERROR] Fast-forward невозможен. Локальные изменения не сбрасывались.
    echo Проверьте git status и историю ветки перед повторной попыткой.
    goto :error
)
echo.
echo [OK] Git-репозиторий успешно обновлён.
goto :post_update

:force_update
echo.
echo [STEP] Получение последних изменений перед принудительным обновлением...
git fetch --prune origin
if errorlevel 1 (
    echo [ERROR] Не удалось получить изменения. Локальные файлы не сбрасывались.
    goto :error
)
echo [WARN] Выполняется git reset --hard origin/%BRANCH%.
git reset --hard origin/%BRANCH%
if errorlevel 1 (
    echo [ERROR] Принудительное обновление завершилось ошибкой.
    goto :error
)
echo [OK] Отслеживаемые файлы обновлены; неотслеживаемые файлы не удалялись.
goto :post_update

:zip_update
echo [INFO] Обнаружена ZIP-копия проекта или недоступен Git.
echo [INFO] Скачиваю архив main и обновляю файлы, не удаляя пользовательские данные.
call :download_archive
if errorlevel 1 (
    echo [ERROR] Не удалось скачать или распаковать архив GitHub.
    call :cleanup_temp
    goto :error
)

robocopy "%TMP_DIR%\Dataset-Curator-main" "%PROJECT_ROOT%" /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XD ".git" ".venv" "venv" "env" "__pycache__" ".idea" ".vscode" "models" ".cache" "selected" "rejected" "bed_size" "blure" "no_or_small_faces" "moree_faces" "duplicate" /XF "dataset_curator_settings.json" "hf_token.txt" "dataset_report.csv" "caption_report.csv" "dataset_curator.sqlite3" "*.db" "*.db-wal" "*.db-shm" "*.sqlite" "*.sqlite3" "*.sqlite-wal" "*.sqlite-shm" "*.sqlite3-wal" "*.sqlite3-shm" "blaze_face_short_range.tflite" "*.log" "*.tmp"
set "ROBOCOPY_RC=!ERRORLEVEL!"
call :cleanup_temp
if !ROBOCOPY_RC! GEQ 8 (
    echo [ERROR] Robocopy завершился с кодом !ROBOCOPY_RC!.
    goto :error
)
echo.
echo [OK] Файлы проекта обновлены из ZIP.
echo [INFO] Устаревшие файлы, отсутствующие в репозитории, автоматически не удалялись.
goto :post_update

:new_install
echo [INFO] Проект в этой папке не найден. Подготовлю отдельную папку Dataset-Curator.
set "TARGET_DIR=%~dp0Dataset-Curator"
if exist "!TARGET_DIR!" (
    echo [ERROR] Папка назначения уже существует:
    echo         !TARGET_DIR!
    echo Переместите её или запустите update.bat из существующей папки проекта.
    goto :error
)
call :locate_git
if not errorlevel 1 goto :clone_with_git
echo [INFO] Git не найден. Выполняю начальную загрузку ZIP-архива.
goto :zip_clone

:clone_with_git
echo [STEP] Клонирование %REPO_URL% ...
git clone --branch "%BRANCH%" "%REPO_URL%" "!TARGET_DIR!"
if errorlevel 1 (
    echo [WARN] Git clone не удался. Пробую ZIP-архив...
    if exist "!TARGET_DIR!" rmdir /s /q "!TARGET_DIR!" >nul 2>&1
    goto :zip_clone
)
echo.
echo [OK] Репозиторий клонирован:
echo      !TARGET_DIR!
echo [NEXT] Откройте папку и запустите start_dataset_curator.bat.
goto :post_new_install

:zip_clone
call :download_archive
if errorlevel 1 (
    echo [ERROR] Не удалось скачать или распаковать архив GitHub.
    call :cleanup_temp
    goto :error
)
mkdir "!TARGET_DIR!" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Не удалось создать папку назначения.
    call :cleanup_temp
    goto :error
)
robocopy "%TMP_DIR%\Dataset-Curator-main" "!TARGET_DIR!" /E /COPY:DAT /DCOPY:DAT /R:2 /W:1 /XD ".git" ".venv" "venv" "env" "__pycache__" ".idea" ".vscode" "models" ".cache" "selected" "rejected" "bed_size" "blure" "no_or_small_faces" "moree_faces" "duplicate" /XF "dataset_curator_settings.json" "hf_token.txt" "dataset_report.csv" "caption_report.csv" "dataset_curator.sqlite3" "*.db" "*.db-wal" "*.db-shm" "*.sqlite" "*.sqlite3" "*.sqlite-wal" "*.sqlite-shm" "*.sqlite3-wal" "*.sqlite3-shm" "blaze_face_short_range.tflite" "*.log" "*.tmp"
set "ROBOCOPY_RC=!ERRORLEVEL!"
call :cleanup_temp
if !ROBOCOPY_RC! GEQ 8 (
    echo [ERROR] Robocopy завершился с кодом !ROBOCOPY_RC!.
    goto :error
)
echo.
echo [OK] ZIP-копия проекта подготовлена:
echo      !TARGET_DIR!
goto :post_new_install

:post_new_install
echo.
echo Для запуска перейдите в папку проекта и откройте start_dataset_curator.bat.
echo Для установки зависимостей используйте requirements.txt; не переустанавливайте PyTorch/CUDA без необходимости.
goto :success

:post_update
echo.
echo Настройки dataset_curator_settings.json, hf_token.txt, модель MediaPipe, отчёты
echo и папки датасета не удалялись. ZIP-обновление не удаляет устаревшие файлы.
echo.
set "RUN_DEPS="
set /p "RUN_DEPS=Обновить Python-зависимости из requirements.txt сейчас? [y/N]: "
if /I "!RUN_DEPS!"=="Y" (
    set "PYTHON=C:\Program Files\Python\python.exe"
    if not exist "!PYTHON!" set "PYTHON=python"
    REM Use the same interpreter as start_dataset_curator.bat; do not let uv select another Python.
    "!PYTHON!" -m pip install -r "%PROJECT_ROOT%\requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Установка зависимостей завершилась с ошибкой.
        goto :error
    )
)
goto :success

:ensure_git
call :locate_git
if not errorlevel 1 exit /b 0
echo [WARN] Git не найден. Пробую установить Git for Windows...
where winget >nul 2>&1
if not errorlevel 1 (
    winget install --id Git.Git -e --source winget --silent --accept-package-agreements --accept-source-agreements
    call :locate_git
    if not errorlevel 1 exit /b 0
)
set "GIT_INSTALLER=%TEMP%\DatasetCurator_GitSetup_%RANDOM%.exe"
echo [STEP] Загружаю официальный установщик Git for Windows...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $h=@{'User-Agent'='Dataset-Curator-Updater'}; $r=Invoke-RestMethod -Uri 'https://api.github.com/repos/git-for-windows/git/releases/latest' -Headers $h; $a=$r.assets | Where-Object { $_.name -match '^Git-.*-64-bit\.exe$' } | Select-Object -First 1; if (-not $a) { throw 'Не найден x64-установщик Git for Windows' }; Invoke-WebRequest -UseBasicParsing -Uri $a.browser_download_url -OutFile $env:GIT_INSTALLER; if (-not (Test-Path -LiteralPath $env:GIT_INSTALLER)) { throw 'Установщик не загружен' }"
if errorlevel 1 exit /b 1
echo [STEP] Установка Git for Windows...
start /wait "" "%GIT_INSTALLER%" /VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS
del /q "%GIT_INSTALLER%" >nul 2>&1
call :locate_git
if not errorlevel 1 exit /b 0
echo [WARN] Не удалось подготовить Git. ZIP-обновление остаётся доступно.
exit /b 1

:locate_git
set "GIT_EXE="
set "GIT_CMD_DIR="
for /f "delims=" %%G in ('where git 2^>nul') do if not defined GIT_EXE set "GIT_EXE=%%G"
if not defined GIT_EXE if exist "%ProgramFiles%\Git\cmd\git.exe" set "GIT_EXE=%ProgramFiles%\Git\cmd\git.exe"
if not defined GIT_EXE if exist "%ProgramFiles(x86)%\Git\cmd\git.exe" set "GIT_EXE=%ProgramFiles(x86)%\Git\cmd\git.exe"
if not defined GIT_EXE if exist "%LOCALAPPDATA%\Programs\Git\cmd\git.exe" set "GIT_EXE=%LOCALAPPDATA%\Programs\Git\cmd\git.exe"
if not defined GIT_EXE exit /b 1
"!GIT_EXE!" --version >nul 2>&1
if errorlevel 1 (
    set "GIT_EXE="
    exit /b 1
)
for %%D in ("!GIT_EXE!") do set "GIT_CMD_DIR=%%~dpD"
set "PATH=!GIT_CMD_DIR!;!PATH!"
exit /b 0

:download_archive
if not exist "%TMP_DIR%" mkdir "%TMP_DIR%" >nul 2>&1
set "DC_UPDATE_TMP=%TMP_DIR%"
set "DC_UPDATE_URL=%ZIP_URL%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $ProgressPreference='SilentlyContinue'; $zip=Join-Path $env:DC_UPDATE_TMP 'repo.zip'; Invoke-WebRequest -UseBasicParsing -Uri $env:DC_UPDATE_URL -OutFile $zip; Expand-Archive -LiteralPath $zip -DestinationPath $env:DC_UPDATE_TMP -Force"
if errorlevel 1 exit /b 1
if not exist "%TMP_DIR%\Dataset-Curator-main\dataset_curator_gui.py" exit /b 1
exit /b 0

:cleanup_temp
if exist "%TMP_DIR%" rmdir /s /q "%TMP_DIR%" >nul 2>&1
exit /b 0

:cancelled
echo.
echo [INFO] Обновление отменено. Локальные файлы не изменялись.
pause
exit /b 0

:success
echo.
echo ============================================================
echo [OK] Операция завершена.
echo ============================================================
pause
exit /b 0

:error
echo.
echo ============================================================
echo [ERROR] Операция не завершена. См. сообщения выше.
echo ============================================================
pause
exit /b 1
