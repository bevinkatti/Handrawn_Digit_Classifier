@echo off
cd /d "%~dp0"
call mnist_env\Scripts\activate.bat
start http://localhost:5000
python app.py