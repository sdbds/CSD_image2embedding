$train_data_dir = "./datasets"
$output_path = "./datasets.lance"
$only_save_path = $false

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if ($Env:OS -eq "Windows_NT") {
    $python = if (Test-Path ".venv/Scripts/python.exe") {
        ".venv/Scripts/python.exe"
    }
    else {
        "venv/Scripts/python.exe"
    }
}
else {
    $python = if (Test-Path ".venv/bin/python") {
        ".venv/bin/python"
    }
    else {
        "venv/bin/python"
    }
}

if (-not (Test-Path $python)) {
    throw "Python environment not found. Run Step1_install-uv.ps1 first."
}

$cliArgs = @($train_data_dir, "--output", $output_path)
if ($only_save_path) {
    $cliArgs += "--only-save-path"
}

& $python -m csd_image2embedding.data @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "Lance snapshot creation failed with exit code $LASTEXITCODE."
}
