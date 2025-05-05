@echo off
echo Starting MongoDB service...
net start MongoDB

echo.
echo Starting Backend...
cd /d "C:\Users\ASMrAbhilash(Abhilas\login-app"
start cmd /k "node server.js"

echo.
echo Starting React Frontend...
cd /d "C:\Users\ASMrAbhilash(Abhilas\login-app"
start cmd /k "npm start"

echo.
echo All services started.
pause
