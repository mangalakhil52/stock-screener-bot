@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt -q
python src\main.py >> logs\daily_run.log 2>&1
