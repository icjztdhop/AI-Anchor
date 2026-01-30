# ===============================
# start_lmdeploy.ps1
# ===============================

[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$RootDir = $PSScriptRoot
Set-Location $RootDir

# ---------- 0) 激活 venv ----------
$VenvActivate = Join-Path $RootDir "venv\Scripts\Activate.ps1"
if (Test-Path $VenvActivate) {
    . $VenvActivate
    Write-Host ("[LMDeploy] venv activated: {0}" -f $env:VIRTUAL_ENV)
} else {
    Write-Error ("[LMDeploy] venv not found: {0}" -f $VenvActivate)
    exit 1
}

# ---------- 1) CUDA_PATH：优先使用工作目录下的 cuda_stub（你已有） ----------
$CudaStub = Join-Path $RootDir "cuda_stub"
if (-not $env:CUDA_PATH -or $env:CUDA_PATH.Trim() -eq "") {
    if (Test-Path $CudaStub) {
        $env:CUDA_PATH = $CudaStub
    }
}
if ($env:CUDA_PATH -and $env:CUDA_PATH.Trim() -ne "") {
    $CudaBin = Join-Path $env:CUDA_PATH "bin"
    if (Test-Path $CudaBin) {
        $env:PATH = "$CudaBin;$env:PATH"
    }
    Write-Host ("[LMDeploy] CUDA_PATH: {0}" -f $env:CUDA_PATH)
}

# ---------- 2) 读取 config.txt ----------
function Load-Config([string]$path) {
    if (!(Test-Path -LiteralPath $path)) {
        Write-Error ("[LMDeploy] config.txt not found: {0}" -f $path)
        exit 1
    }
    $cfg = @{}
    Get-Content -LiteralPath $path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }
        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        if (($val.StartsWith('"') -and $val.EndsWith('"')) -or ($val.StartsWith("'") -and $val.EndsWith("'"))) {
            $val = $val.Substring(1, $val.Length - 2)
        }
        $cfg[$key] = $val
        Set-Item -Path ("Env:{0}" -f $key) -Value $val
    }
    return $cfg
}

$CfgPath = Join-Path $RootDir "config.txt"
$CFG = Load-Config $CfgPath

function Cfg([string]$k, [string]$d="") {
    if ($CFG.ContainsKey($k) -and $CFG[$k].Trim() -ne "") { return $CFG[$k].Trim() }
    return $d
}

# ---------- 3) 从 config 取参数 ----------
$Subcmd          = Cfg "LMDEPLOY_SUBCMD" "serve api_server"   # 你给的是 "serve api_server"
$ModelRel        = Cfg "LMDEPLOY_MODEL_PATH" ""
$Backend         = Cfg "LMDEPLOY_BACKEND" "turbomind"
$ModelFormat     = Cfg "LMDEPLOY_MODEL_FORMAT" ""            # 例如 awq
$ModelName       = Cfg "LMDEPLOY_MODEL_NAME" ""              # 例如 internlm2_5_7b_chat_awq_int4
$ServerName      = Cfg "LMDEPLOY_SERVER_NAME" "0.0.0.0"
$ServerPort      = Cfg "LMDEPLOY_SERVER_PORT" "23333"
$TP              = Cfg "LMDEPLOY_TP" ""
$SessionLen      = Cfg "LMDEPLOY_SESSION_LEN" ""
$CacheMaxEntry   = Cfg "LMDEPLOY_CACHE_MAX_ENTRY_COUNT" ""

# ---------- 4) 强制本地模型路径（关键：避免当成在线 repo id） ----------
if ($ModelRel -eq "") {
    Write-Error "[LMDeploy] LMDEPLOY_MODEL_PATH is empty in config.txt"
    exit 1
}

$ModelPath = $ModelRel
if (-not [System.IO.Path]::IsPathRooted($ModelPath)) {
    $ModelPath = Join-Path $RootDir $ModelPath
}

Write-Host "==========================================="
Write-Host ("[LMDeploy] RootDir      : {0}" -f $RootDir)
Write-Host ("[LMDeploy] Subcmd       : {0}" -f $Subcmd)
Write-Host ("[LMDeploy] ModelPath    : {0}" -f $ModelPath)
Write-Host ("[LMDeploy] Backend      : {0}" -f $Backend)
Write-Host ("[LMDeploy] ModelFormat  : {0}" -f $ModelFormat)
Write-Host ("[LMDeploy] ModelName    : {0}" -f $ModelName)
Write-Host ("[LMDeploy] ServerName   : {0}" -f $ServerName)
Write-Host ("[LMDeploy] ServerPort   : {0}" -f $ServerPort)
Write-Host ("[LMDeploy] TP           : {0}" -f $TP)
Write-Host ("[LMDeploy] SessionLen   : {0}" -f $SessionLen)
Write-Host ("[LMDeploy] CacheMaxEntry: {0}" -f $CacheMaxEntry)
Write-Host "==========================================="

# 关键：本地目录不存在就直接失败，阻止 lmdeploy 去“查在线模型”
if (!(Test-Path -LiteralPath $ModelPath)) {
    Write-Error ("[LMDeploy] 本地模型路径不存在（将不会尝试在线下载）：{0}" -f $ModelPath)
    $ModelDir = Join-Path $RootDir "model"
    if (Test-Path -LiteralPath $ModelDir) {
        Write-Host "[Hint] 当前 .\model\ 下的条目："
        Get-ChildItem -LiteralPath $ModelDir | ForEach-Object { Write-Host ("  - {0}" -f $_.Name) }
    }
    exit 1
}

# ---------- 5) 进一步阻止在线访问（可选但推荐） ----------
# 如果你确实想让它联网下载，把下面三行注释掉即可
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"

# ---------- 6) 拼 lmdeploy 命令 ----------
# 你的版本是：lmdeploy serve api_server model_path --server-name --server-port ...
$SubParts = $Subcmd.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)

$LmArgs = @()
$LmArgs += $SubParts
$LmArgs += @($ModelPath)
$LmArgs += @("--server-name", $ServerName, "--server-port", $ServerPort)

# 后端/格式/名称/并行/上下文/cache
if ($Backend -ne "")       { $LmArgs += @("--backend", $Backend) }
if ($ModelFormat -ne "")   { $LmArgs += @("--model-format", $ModelFormat) }
if ($ModelName -ne "")     { $LmArgs += @("--model-name", $ModelName) }
if ($TP -ne "")            { $LmArgs += @("--tp", $TP) }
if ($SessionLen -ne "")    { $LmArgs += @("--session-len", $SessionLen) }
if ($CacheMaxEntry -ne "") { $LmArgs += @("--cache-max-entry-count", $CacheMaxEntry) }

# 建议给个 log-level（你也可以写进 config 再读）
if (-not $CFG.ContainsKey("LMDEPLOY_LOG_LEVEL")) {
    $LmArgs += @("--log-level", "WARNING")
} else {
    $LmArgs += @("--log-level", (Cfg "LMDEPLOY_LOG_LEVEL" "WARNING"))
}

Write-Host ("[LMDeploy] Command: lmdeploy {0}" -f ($LmArgs -join " "))

# ---------- 7) 启动 ----------
& lmdeploy @LmArgs
exit $LASTEXITCODE
