$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not $Env:HF_HOME) {
    $Env:HF_HOME = Join-Path $PSScriptRoot "huggingface"
}
if (-not $Env:HF_ENDPOINT) {
    $Env:HF_ENDPOINT = "https://hf-mirror.com"
}
if (-not $Env:UV_INDEX_URL) {
    $Env:UV_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple/"
}
if (-not $Env:UV_EXTRA_INDEX_URL) {
    $Env:UV_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cu130"
}
$Env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
$Env:UV_LINK_MODE = "copy"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Output "Installing uv..."
    if ($Env:OS -eq "Windows_NT") {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    }
    else {
        sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"
    }
    $uvBin = Join-Path $HOME ".local/bin"
    $Env:PATH = "$uvBin$([IO.Path]::PathSeparator)$Env:PATH"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was installed but is not available in this shell. Open a new shell and rerun this script."
}

if (-not (Test-Path ".venv")) {
    uv venv .venv --python 3.11
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create the Python 3.11 environment."
    }
}

Write-Output "Installing locked runtime dependencies..."
uv pip sync requirements-uv.txt --python .venv --index-strategy unsafe-best-match
if ($LASTEXITCODE -ne 0) {
    throw "Runtime dependency installation failed."
}

Write-Output "Trying the optional accelerated KMeans backend..."
if ($Env:OS -eq "Windows_NT") {
    uv pip install triton-windows --python .venv
}
else {
    uv pip install triton --python .venv
}

if ($LASTEXITCODE -eq 0) {
    uv pip install --no-deps `
        "git+https://github.com/svg-project/flash-kmeans.git@main" `
        --python .venv
}

if ($LASTEXITCODE -ne 0) {
    Write-Warning "flash-kmeans is unavailable; scikit-learn KMeans will be used."
}

Write-Output "Installation finished."
