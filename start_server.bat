@echo off
cd /d %~dp0
echo Starting AI ETF Trader Web Server...
start "AI Trader Web" python src/web_app.py
echo.
echo Web Server started in a new window.
echo.
echo Starting Daily Task Scheduler (Keep this window open)...
python src/main.py
pause
