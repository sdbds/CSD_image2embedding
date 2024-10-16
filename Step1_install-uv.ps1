Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
#$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"
$Env:UV_CACHE_DIR="./.cache"
$Env:UV_NO_CACHE=0
$Env:UV_LINK_MODE="copy"
$Env:FAISS_ENABLE_GPU="ON"
function InstallFail {
    Write-Output "Install failed。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

try {
    ~/.cargo/bin/uv --version
    ~/.cargo/bin/uv self update
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Installing uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "./uv-installer.ps1"
        Check "Install failed|安装uv模块失败。"
    }
    else {
        sh "./uv-installer.sh"
        Check "Install failed|安装uv模块失败。"
    }
}

if ($env:OS -ilike "*windows*") {
    if (Test-Path "./venv/Scripts/activate") {
        Write-Output "Windows venv"
        . ./venv/Scripts/activate
    }
    elseif (Test-Path "./.venv/Scripts/activate") {
        Write-Output "Windows .venv"
        . ./.venv/Scripts/activate
    }else{
        Write-Output "Create .venv"
        ~\.cargo\bin\uv.exe venv -p 3.10
        . ./.venv/Scripts/activate
    }
}
elseif (Test-Path "./venv/bin/activate") {
    Write-Output "Linux venv"
    . ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
    Write-Output "Linux .venv"
    . ./.venv/bin/activate.ps1
}
else{
    Write-Output "Create .venv"
    ~/.cargo/bin/uv venv -p 3.10
    . ./.venv/bin/activate.ps1
}

Write-Output "Requirements installing|安装程序所需依赖"

~/.cargo/bin/uv pip sync ./requirements-uv.txt --index-strategy unsafe-best-match
Check "Requirements install failed|环境安装失败。"

Write-Output "Clean cache"
~/.cargo/bin/uv cache clean

Write-Output "Install finished"
Read-Host | Out-Null ;
