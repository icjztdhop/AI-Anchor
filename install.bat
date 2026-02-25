@echo off
chcp 65001 > nul
setlocal EnableExtensions EnableDelayedExpansion

echo ===============================
echo AI-Anchor Windows 安装脚本
echo ===============================

set "PYEXE="

REM --------------------------------
REM 1) 自动检测：遍历 where python 的所有结果，找第一个能运行的
REM --------------------------------
for /f "delims=" %%i in ('where python 2^>nul') do (
    "%%i" --version >nul 2>&1
    if !errorlevel! == 0 (
        set "PYEXE=%%i"
        goto :python_ok
    )
)

REM --------------------------------
REM 2) 自动检测失败：提示输入（循环直到输入正确）
REM --------------------------------
:ask_python
echo.
echo 未检测到可用的 Python（可能是 Windows Store 的 python 别名）。

echo 请输入 python.exe 的完整路径（可拖拽 python.exe 到此窗口回车）：
set /p "PYEXE=> "

REM 去掉引号
set "PYEXE=%PYEXE:"=%"

if not exist "%PYEXE%" (
    echo ❌ 路径不存在，请重新输入。
    goto :ask_python
)

"%PYEXE%" --version >nul 2>&1
if errorlevel 1 (
    echo ❌ 该文件无法运行（可能不是 python.exe），请重新输入。
    goto :ask_python
)

:python_ok
for /f "delims=" %%v in ('"%PYEXE%" --version 2^>^&1') do set "PYVER=%%v"

echo.
echo ✔ 使用 Python：%PYEXE%
echo ✔ %PYVER%

REM --------------------------------
REM 3) 创建虚拟环境
REM --------------------------------
if not exist venv (
    echo ▶ 创建虚拟环境...
    "%PYEXE%" -m venv venv
    if errorlevel 1 (
        echo ❌ 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo ✔ 虚拟环境已存在，跳过创建
)

REM --------------------------------
REM 4) 激活虚拟环境
REM --------------------------------
echo ▶ 激活虚拟环境...
call venv\Scripts\activate
if errorlevel 1 (
    echo ❌ 激活虚拟环境失败
    pause
    exit /b 1
)

REM --------------------------------
REM 5) 升级 pip
REM --------------------------------
echo ▶ 升级 pip...
python -m pip install --upgrade pip

REM --------------------------------
REM 6) 安装依赖
REM --------------------------------
if exist requirements.txt (
    echo ▶ 安装依赖 requirements.txt...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
) else (
    echo ❌ 未找到 requirements.txt
)

REM --------------------------------
REM 7) 创建目录
REM --------------------------------
echo ▶ 创建必要目录...
if not exist Live2D mkdir Live2D
if not exist model mkdir model

echo.
echo ✅ 安装完成！
pause
endlocal