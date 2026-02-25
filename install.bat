@echo off
chcp 65001 > nul

echo ===============================
echo AI-Anchor Windows 安装脚本
echo ===============================

REM 检查 Python
C:\Users\SHAWN.FU\Desktop\AI-Anchor\Python\python.exe --version > nul 2>&1
if errorlevel 1 (
    echo ❌ 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b
)

REM 创建虚拟环境
if not exist venv (
    echo ▶ 创建虚拟环境...
    C:\Users\SHAWN.FU\Desktop\AI-Anchor\Python\python.exe -m venv venv
) else (
    echo ✔ 虚拟环境已存在，跳过创建
)

REM 激活虚拟环境
echo ▶ 激活虚拟环境...
call venv\Scripts\activate

REM 升级 pip
echo ▶ 升级 pip...
python -m pip install --upgrade pip

REM 安装依赖
if exist requirements.txt (
    echo ▶ 安装依赖 requirements.txt...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo ❌ 未找到 requirements.txt
)

REM 创建目录
echo ▶ 创建必要目录...
if not exist Live2D mkdir Live2D
if not exist model mkdir model

echo.
echo ✅ 安装完成！
echo 👉 已创建：
echo    - venv 虚拟环境
echo    - Live2D 文件夹
echo    - model 文件夹
echo.
pause
