@echo off
rem Double-click launcher for Mali Agent.
rem
rem Starts the local model server, starts the dashboard, opens the browser. Nothing
rem here reaches the network: Foundry Local and the web server both bind 127.0.0.1.
rem
rem Kept as a .cmd rather than a compiled exe on purpose -- it is readable, it needs no
rem build step, and it keeps working when the code changes. See README for how to wrap
rem it as a real .exe if you want a desktop icon without the console window.

setlocal
cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo.
    echo Sanal ortam bulunamadi: %PY%
    echo Once kurulumu yapin:
    echo     python -m venv .venv
    echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

rem --db is deliberately explicit. Without it the app opens faturalar.db, and which
rem database is in use should never be a surprise on a double-click.
set "DB=%~dp0faturalar.db"
if not "%~1"=="" set "DB=%~1"

echo Mali Agent baslatiliyor...
echo   veritabani: %DB%
echo.

"%PY%" main.py --serve --db "%DB%" --open-browser --port 8000

rem Reached when the server stops. Without the pause a crash would close the window
rem before the traceback could be read.
echo.
echo Sunucu durdu.
pause
