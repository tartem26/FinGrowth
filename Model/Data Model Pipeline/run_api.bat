:: Optional Windows helper
@echo off
cd /d "%~dp0"
python -m uvicorn predict_api:app --host 127.0.0.1 --port 5055 --reload
pause