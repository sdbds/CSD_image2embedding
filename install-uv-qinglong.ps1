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
    Write-Output "��װʧ�ܡ�"
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
    Write-Output "uv installed|UVģ���Ѱ�װ."
}
catch {
    Write-Output "Install uv|��װuvģ����..."
    if ($Env:OS -ilike "*windows*") {
        powershell -ExecutionPolicy ByPass -c "./uv-installer.ps1"
        Check "��װuvģ��ʧ�ܡ�"
    }
    else {
        sh "./uv-installer.sh"
        Check "��װuvģ��ʧ�ܡ�"
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

Write-Output "��װ������������ (�ѽ��й��ڼ��٣����ڹ�����޷�ʹ�ü���Դ�뻻�� install.ps1 �ű�)"

~/.cargo/bin/uv pip sync ./requirements-uv.txt
Check "������װʧ�ܡ�"

# $download_hy = Read-Host "�Ƿ�����CSDģ��? ����Ҫ����ģ��ѡ�� y ��������Ҫѡ�� n��[y/n] (Ĭ��Ϊ n)"
# if ($download_hy -eq "y" -or $download_hy -eq "Y"){
#     huggingface-cli download tomg-group-umd/CSD-ViT-L --local-dir ./pretrainedmodels
# }

Write-Output "��װ���"
Read-Host | Out-Null ;
