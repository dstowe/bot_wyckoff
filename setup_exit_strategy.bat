@echo off
chcp 65001 >nul
echo 🎯 Enhanced Exit Strategy Refinement - Complete Setup
echo ========================================================
echo.

echo Step 1: Testing the Enhanced Exit Strategy System...
python test_exit_strategy.py

echo.
echo Step 2: Integrating into your existing bot...
python integrate_exit_strategy.py

echo.
echo ✅ Enhanced Exit Strategy Setup Complete!
echo.
echo 📋 What was implemented:
echo    • Dynamic profit targets based on VIX volatility
echo    • Low VIX (^<20): Extended targets (6%% → 8%%, 12%% → 16%%, etc.)
echo    • High VIX (^>25): Tighter targets (6%% → 4%%, 12%% → 8%%, etc.)
echo    • Position-specific adjustments for time held and Wyckoff phase
echo    • Automatic integration with day trade protection
echo.
echo 🚀 Your bot now has sophisticated exit strategies!
echo    Run your normal trading bot to see enhanced exits in action.
echo.
pause
