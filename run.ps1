# =========================================
# run.ps1
# 一键启动：LMDeploy + GPT-SoVITS + API（同一套逻辑）
# - LMDeploy / GPT-SoVITS：后台启动，日志写入 lmdeploy.log / gptsovits.log
# - API：前台运行（按 Ctrl+C 退出）
# - 退出时自动停止后台进程
# =========================================

$ErrorActionPreference = "Stop"

function Load-Config([string]$path) {
    $cfg = @{}
    Get-Content -LiteralPath $path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#") -or $line.StartsWith(";")) { return }
        if ($line -match "^(.*?)=(.*)$") {
            $cfg[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    return $cfg
}

$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$CFG_PATH = Join-Path $ROOT "config.txt"

if (-not (Test-Path -LiteralPath $CFG_PATH)) {
    Write-Error "config.txt not found in: $ROOT"
    exit 1
}

$CFG = Load-Config $CFG_PATH

Write-Host "========================================="
Write-Host " Virtual Anchor - Start All (PowerShell)"
Write-Host "========================================="

# Export config to env (Python / child processes can read)
foreach ($k in $CFG.Keys) { $env:$k = $CFG[$k] }

function Require([string]$name) {
    if (-not $CFG.ContainsKey($name) -or [string]::IsNullOrWhiteSpace($CFG[$name])) {
        throw "Missing config key: $name"
    }
}

@(
  "LMDEPLOY_SUBCMD","LMDEPLOY_MODEL_PATH","LMDEPLOY_BACKEND","LMDEPLOY_MODEL_FORMAT","LMDEPLOY_MODEL_NAME",
  "LMDEPLOY_SERVER_NAME","LMDEPLOY_SERVER_PORT","LMDEPLOY_TP","LMDEPLOY_SESSION_LEN","LMDEPLOY_CACHE_MAX_ENTRY_COUNT",
  "GPTSOVITS_DIR","GPTSOVITS_API","GPTSOVITS_HOST","GPTSOVITS_PORT","API_ENTRY"
) | ForEach-Object { Require $_ }

$lmdeployLog = Join-Path $ROOT "lmdeploy.log"
$gptsovitsLog = Join-Path $ROOT "gptsovits.log"

$lmProc = $null
$ttsProc = $null

try {
    Write-Host "[1/3] Starting LMDeploy..."
    $lmArgs = @(
        $CFG["LMDEPLOY_SUBCMD"],
        $CFG["LMDEPLOY_MODEL_PATH"],
        "--backend", $CFG["LMDEPLOY_BACKEND"],
        "--model-format", $CFG["LMDEPLOY_MODEL_FORMAT"],
        "--model-name", $CFG["LMDEPLOY_MODEL_NAME"],
        "--server-name", $CFG["LMDEPLOY_SERVER_NAME"],
        "--server-port", $CFG["LMDEPLOY_SERVER_PORT"],
        "--tp", $CFG["LMDEPLOY_TP"],
        "--session-len", $CFG["LMDEPLOY_SESSION_LEN"],
        "--cache-max-entry-count", $CFG["LMDEPLOY_CACHE_MAX_ENTRY_COUNT"]
    )

    $lmProc = Start-Process -FilePath "lmdeploy" -ArgumentList $lmArgs `
        -WorkingDirectory $ROOT -RedirectStandardOutput $lmdeployLog -RedirectStandardError $lmdeployLog `
        -NoNewWindow -PassThru

    Start-Sleep -Seconds 5

    Write-Host "[2/3] Starting GPT-SoVITS..."
    $ttsWorkDir = Join-Path $ROOT $CFG["GPTSOVITS_DIR"]

    $ttsArgs = @(
        $CFG["GPTSOVITS_API"],
        "--host", $CFG["GPTSOVITS_HOST"],
        "--port", $CFG["GPTSOVITS_PORT"]
    )
    if ($CFG.ContainsKey("GPTSOVITS_ARGS") -and -not [string]::IsNullOrWhiteSpace($CFG["GPTSOVITS_ARGS"])) {
        $ttsArgs += ($CFG["GPTSOVITS_ARGS"] -split "\s+")
    }

    $ttsProc = Start-Process -FilePath "python" -ArgumentList $ttsArgs `
        -WorkingDirectory $ttsWorkDir -RedirectStandardOutput $gptsovitsLog -RedirectStandardError $gptsovitsLog `
        -NoNewWindow -PassThru

    Start-Sleep -Seconds 5

    Write-Host "[3/3] Starting API Server..."
    Push-Location $ROOT
    try {
        & python $CFG["API_ENTRY"]
    } finally {
        Pop-Location
    }
}
finally {
    Write-Host ""
    Write-Host "Stopping services..."
    if ($lmProc -and -not $lmProc.HasExited) { Stop-Process -Id $lmProc.Id -Force -ErrorAction SilentlyContinue }
    if ($ttsProc -and -not $ttsProc.HasExited) { Stop-Process -Id $ttsProc.Id -Force -ErrorAction SilentlyContinue }
}
