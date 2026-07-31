$train_data_dir = "./datasets"
$dataset_path = $null # Set to an existing Lance directory to bypass discovery.
$artifact_root = "./.artifacts"
$output_dir = "./output"

$backend = "csd" # csd or siglip-dinov3
$text_mode = "image-only" # image-only or caption-guided
$style_model_config = "./configs/siglip_dinov3.yaml"
$style_model_checkpoint = $null

$batch_size = 24
$k_clusters = 40
$min_cluster_size = 10
$finch_partition_index = 1
$precision = "fp16" # auto, fp32, fp16, or bf16
$reducer = $null # pacmap, umap, tsne, ivis, or legacy
$symlink = $true
$rebuild = $false

# Do not modify below this line unless changing the launch behavior.
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

if (-not $Env:HF_HOME) {
    $Env:HF_HOME = Join-Path $PSScriptRoot "huggingface"
}
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"

$cliArgs = @(
    "--train-data-dir", $train_data_dir,
    "--artifact-root", $artifact_root,
    "--output-dir", $output_dir,
    "--backend", $backend,
    "--text-mode", $text_mode,
    "--batch-size", $batch_size,
    "--k-clusters", $k_clusters,
    "--min-cluster-size", $min_cluster_size,
    "--finch-partition-index", $finch_partition_index,
    "--precision", $precision
)

if ($dataset_path) {
    $cliArgs += @("--dataset-path", $dataset_path)
}
if ($backend -eq "siglip-dinov3") {
    $cliArgs += @("--style-model-config", $style_model_config)
}
if ($style_model_checkpoint) {
    $cliArgs += @("--style-model-checkpoint", $style_model_checkpoint)
}
if ($reducer) {
    $cliArgs += @("--reducer", $reducer)
}
if ($symlink) {
    $cliArgs += "--symlink"
}
if ($rebuild) {
    $cliArgs += "--rebuild"
}

& $python -m csd_image2embedding @cliArgs
if ($LASTEXITCODE -ne 0) {
    throw "Embedding workflow failed with exit code $LASTEXITCODE."
}
