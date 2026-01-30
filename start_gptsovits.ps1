# ===============================
# GPT-SoVITS 启动脚本
# 配置来源：config.txt
# ===============================

$RootDir  = $PSScriptRoot
$CfgFile  = Join-Path $RootDir "config.txt"

function Load-ConfigFile($path) {
    if (!(Test-Path $path)) {
        Write-Error "❌ config.txt not found: $path"
        exit 1
    }

    Get-Content $path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq "" -or $line.StartsWith("#")) { return }

        $idx = $line.IndexOf("=")
        if ($idx -lt 1) { return }

        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()

        # 去掉引号
        if (($val.StartsWith('"') -and $val.EndsWith('"')) -or
            ($val.StartsWith("'") -and $val.EndsWith("'"))) {
            $val = $val.Substring(1, $val.Length - 2)
        }

        Set-Item -Path "Env:$key" -Value $val
    }
}


Load-ConfigFile $CfgFile


if (-not $env:GPTSOVITS_DIR) { $env:GPTSOVITS_DIR = "GPT-SoVITS" }
if (-not $env:GPTSOVITS_API) { $env:GPTSOVITS_API = "GPT-SoVITS\api_v2.py" }
if (-not $env:GPTSOVITS_PORT) { $env:GPTSOVITS_PORT = "9880" }


$GptDir    = Join-Path $RootDir $env:GPTSOVITS_DIR
$PythonExe = Join-Path $GptDir "runtime\python.exe"
$ApiScript = Join-Path $RootDir $env:GPTSOVITS_API


if ($env:TTS_REF_AUDIO_PATH) {
    $env:TTS_REF_AUDIO_PATH = (Resolve-Path (Join-Path $RootDir $env:TTS_REF_AUDIO_PATH)).Path
}


if (-not $env:GPTSOVITS_URL) {
    $env:GPTSOVITS_URL = "http://127.0.0.1:$($env:GPTSOVITS_PORT)"
}

Write-Host "==========================================="
Write-Host "GPT-SoVITS Config (from config.txt)"
Write-Host "RootDir : $RootDir"
Write-Host "GptDir  : $GptDir"
Write-Host "Python  : $PythonExe"
Write-Host "API     : $ApiScript"
Write-Host "URL     : $($env:GPTSOVITS_URL)"
Write-Host "==========================================="


if (!(Test-Path $PythonExe)) { Write-Error "❌ python.exe not found"; exit 1 }
if (!(Test-Path $ApiScript)) { Write-Error "❌ api_v2.py not found"; exit 1 }


Set-Location $GptDir


$env:PATH = "$GptDir\runtime;$env:PATH"


$Args = @()
if ($env:GPTSOVITS_ARGS -and $env:GPTSOVITS_ARGS.Trim() -ne "") {
    $Args += $env:GPTSOVITS_ARGS.Split(" ", [System.StringSplitOptions]::RemoveEmptyEntries)
}


if ($env:GPTSOVITS_REDIRECT_LOG -eq "1") {
    $LogDir = Join-Path $RootDir "logs"
    if (!(Test-Path $LogDir)) { New-Item -ItemType Directory $LogDir | Out-Null }
    $LogFile = Join-Path $LogDir ("gptsovits_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

    & $PythonExe -I $ApiScript @Args *>> $LogFile
} else {
    & $PythonExe -I $ApiScript @Args
}