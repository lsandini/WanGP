@echo off
title WanGP
cd /d "%~dp0"
call "%~dp0env\Scripts\activate.bat"

:run_loop
echo [*] Launching WanGP - the browser will open when it is ready...
python wgp.py --multiple-images --advanced --open-browser
if %errorlevel% equ 42 (
    echo [*] Restarting...
    goto run_loop
)
pause
