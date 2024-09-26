Set-Location $PSScriptRoot

$Env:HF_HOME = "huggingface"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
# $Env:UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple/"
$Env:UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"
$Env:UV_LINK_MODE="symlink"
$Env:FAISS_ENABLE_GPU="ON"
function InstallFail {
    Write-Output "安装失败。"
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
    ~\.cargo\bin\uv --version
    Write-Output "uv installed|UV模块已安装."
}
catch {
    Write-Output "Install uv|安装uv模块中..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "./uv-installer.ps1"
        Check "安装uv模块失败。"
    }
    else {
        sh "./uv-installer.sh"
        Check "安装uv模块失败。"
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

Write-Output "安装程序所需依赖 (已进行国内加速，若在国外或无法使用加速源请换用 install.ps1 脚本)"

~/.cargo/bin/uv pip sync ./requirements-uv.txt
Check "环境安装失败。"

# $download_hy = Read-Host "是否下载CSD模型? 若需要下载模型选择 y ，若不需要选择 n。[y/n] (默认为 n)"
# if ($download_hy -eq "y" -or $download_hy -eq "Y"){
#     huggingface-cli download tomg-group-umd/CSD-ViT-L --local-dir ./pretrainedmodels
# }

Write-Output "安装完毕"
Read-Host | Out-Null ;
