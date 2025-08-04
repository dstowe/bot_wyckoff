@echo off
REM Enhanced Wyckoff Trading Bot Runner for Windows Task Scheduler
REM Run this batch file with Task Scheduler to execute the enhanced trading bot

echo ============================================
echo Enhanced Wyckoff Trading Bot Starting...
echo Time: %date% %time%
echo ============================================

REM Change to the directory where your trading bot is located
REM Update this path to match your actual installation directory
cd /d "C:\path\to\your\bot"

REM Activate virtual environment if you're using one
REM Uncomment and modify the path below if you have a virtual environment
REM call venv\Scripts\activate.bat

REM Run the enhanced trading bot
python enhanced_wyckoff_trading_bot.py

REM Check the exit code
if %ERRORLEVEL% EQU 0 (
    echo ============================================
    echo Enhanced Trading Bot Completed Successfully
    echo Time: %date% %time%
    echo ============================================
) else (
    echo ============================================
    echo Enhanced Trading Bot Failed with Error Code: %ERRORLEVEL%
    echo Time: %date% %time%
    echo Check logs for details
    echo ============================================
)

REM Optional: Generate analytics report after successful run
if %ERRORLEVEL% EQU 0 (
    echo Generating analytics report...
    python enhanced_trading_analytics.py --report --charts
)

REM Pause for 5 seconds to see the result (remove for production)
timeout /t 5 /nobreak > nul

exit %ERRORLEVEL%