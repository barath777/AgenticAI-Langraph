@echo off
setlocal

:: Force Administrator mode
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting Administrator privileges...
    powershell -Command "Start-Process cmd -ArgumentList '/k \"%~f0\"' -Verb RunAs"
    exit /b
)

echo.
echo *** Restarting MongoDB Windows Service ***
echo.

:: STOP the service
echo Stopping MongoDB service...
net stop MongoDB
echo %errorlevel% after stopping MongoDB

:: Wait a few seconds
timeout /t 5 /nobreak

:: START the service
echo Starting MongoDB service...
net start MongoDB
echo %errorlevel% after starting MongoDB

echo.
echo *** MongoDB Windows Service restart process completed ***
echo.
pause

endlocal
