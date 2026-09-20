@echo off
title E-Consultation Sentiment Analysis

cd /d "%~dp0"

echo Starting E-Consultation Sentiment Analysis...
echo.

start "E-Consultation Flask App" cmd /k ""%~dp0venv\Scripts\python.exe" "%~dp0app.py""

echo Waiting for Flask server...
timeout /t 5 /nobreak >nul

echo Opening application...
start "" "http://127.0.0.1:5000/"

exit
