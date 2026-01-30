@echo off
setlocal

cd /d "%~dp0"

echo =========================================
echo  Virtual Anchor - Start All
echo =========================================

if not exist "config.txt" (
  echo [ERROR] config.txt not found
  exit /b 1
)

:: 如果你希望窗口不要自动关，把 -NoExit 保留
:: 如果希望脚本结束就关窗口，把 -NoExit 去掉

echo [1/3] Starting LMDeploy...
start "LMDeploy" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0start_lmdeploy.ps1"

timeout /t 3 >nul

echo [2/3] Starting GPT-SoVITS...
start "GPT-SoVITS" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0start_gptsovits.ps1"

timeout /t 3 >nul

echo [3/3] Starting API Server...
start "API Server" powershell -NoExit -ExecutionPolicy Bypass -File "%~dp0start_api.ps1"

echo =========================================
echo  All services launched (in separate windows)
echo =========================================
endlocal
