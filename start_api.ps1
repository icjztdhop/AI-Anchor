# ===============================
# start_api.ps1
# ===============================

# -------- 0) 计算根目录 --------
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path

# -------- 1) 激活虚拟环境 --------
$VenvActivate = Join-Path $ROOT "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
    Write-Host "[API] venv activated: $env:VIRTUAL_ENV"
} else {
    Write-Error "[API] venv not found: $VenvActivate"
    exit 1
}

# -------- 2) Load config --------
function Load-Config($path) {
    $cfg = @{}
    Get-Content $path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        if ($line -match "^(.*?)=(.*)$") {
            $cfg[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    return $cfg
}

$CFG = Load-Config (Join-Path $ROOT "config.txt")

Write-Host "[API] Starting FastAPI on $($CFG.API_HOST):$($CFG.API_PORT)"

# -------- 3) 导出环境变量 --------
$env:API_HOST      = $CFG.API_HOST
$env:API_PORT      = $CFG.API_PORT
$env:API_RELOAD    = $CFG.API_RELOAD
$env:API_LOG_LEVEL = $CFG.API_LOG_LEVEL

# -------- 4) 启动 FastAPI --------
# 这里用 python，是 venv 里的 python（因为已经 Activate）
python $CFG.API_ENTRY
