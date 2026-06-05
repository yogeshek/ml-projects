@echo off
echo ===============================================
echo   SMS Spam Detection API - Starting Server
echo ===============================================
echo.

cd /d "%~dp0"

echo Checking if models exist...
if not exist "..\..\..\..\ml-projects\models\vectorizer.pkl" (
    echo [ERROR] Models not found! Please train models first:
    echo   python train_sms.py
    echo.
    pause
    exit /b 1
)

echo [OK] Models found!
echo.
echo Starting FastAPI server...
echo   - API Docs: http://localhost:8000/docs
echo   - Health Check: http://localhost:8000/health
echo   - Stop server: Press Ctrl+C
echo.

python api.py

pause
