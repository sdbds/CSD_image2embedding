Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu130"
#$Env:UV_CACHE_DIR="./.cache"
$Env:UV_NO_CACHE=0
$Env:UV_LINK_MODE="copy"
$Env:FAISS_ENABLE_GPU="ON"
function InstallFail {
    Write-Output "Install failed��"
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
    uv --version
    Write-Output "uv installed|UVģ���Ѱ�װ."
}
catch {
    Write-Output "Installing uv|��װuvģ����..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        Check "Install failed|��װuvģ��ʧ�ܡ�"
    }
    else {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        Check "Install failed|��װuvģ��ʧ�ܡ�"
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
        uv venv -p 3.11
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
    uv venv -p 3.11
    . ./.venv/bin/activate.ps1
}

Write-Output "Requirements installing|��װ������������"

uv pip sync ./requirements-uv.txt --index-strategy unsafe-best-match
Check "Requirements install failed|������װʧ�ܡ�"

Write-Output "Install finished"
Read-Host | Out-Null ;
