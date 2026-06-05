@echo off
title Project Change Logger
echo Starting Project Change Logger...
where pythonw >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    start /B pythonw "%~dp0change_logger.py"
    echo Logger started in windowless mode (background process).
) else (
    start "Project Change Logger" python "%~dp0change_logger.py"
    echo Logger started in a new command window.
)
echo Logging activity to project_changes.log
pause
