$batch_size = 24
$k_clusters = 40 # k_clusters for K clustering
$min_cluster_size = 10 # min_cluster_size for auto clustering
$symlink = 1 # 0 or 1 for False or True about use symlink
$precision = "fp16" # auto, fp32, fp16, bf16
$model_type = "csd" # csd or sd
$finch_partition_index = 1 # FINCH hierarchy partition index
$sd_config = "sd_config.yaml" # used when model_type = sd
$sd_checkpoint = $null # optional checkpoint override for sd

# ============= DO NOT MODIFY CONTENTS BELOW | 请勿修改下方内容 =====================
# Activate python venv
Set-Location $PSScriptRoot
if ($env:OS -ilike "*windows*") {
  if (Test-Path "./venv/Scripts/activate") {
    Write-Output "Windows venv"
    ./venv/Scripts/activate
  }
  elseif (Test-Path "./.venv/Scripts/activate") {
    Write-Output "Windows .venv"
    ./.venv/Scripts/activate
  }
}
elseif (Test-Path "./venv/bin/activate") {
  Write-Output "Linux venv"
  ./venv/bin/Activate.ps1
}
elseif (Test-Path "./.venv/bin/activate") {
  Write-Output "Linux .venv"
  ./.venv/bin/activate.ps1
}

$Env:HF_HOME = "huggingface"
$Env:XFORMERS_FORCE_DISABLE_TRITON = "1"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
$ext_args = [System.Collections.ArrayList]::new()

if ($batch_size -ne 12) {
  [void]$ext_args.Add("--batch_size=$batch_size")
}

if ($k_clusters -ne 40) {
  [void]$ext_args.Add("--k_clusters=$k_clusters")
}

if ($min_cluster_size -ne 10) {
  [void]$ext_args.Add("--min_cluster_size=$min_cluster_size")
}

if ($symlink) {
  [void]$ext_args.Add("--symlink")
}

if ($precision -ne "auto") {
  [void]$ext_args.Add("--precision=$precision")
}

if ($model_type -ne "csd") {
  [void]$ext_args.Add("--model_type=$model_type")
}

if ($finch_partition_index -ne 1) {
  [void]$ext_args.Add("--finch_partition_index=$finch_partition_index")
}

if ($sd_config -ne "sd_config.yaml") {
  [void]$ext_args.Add("--sd_config=$sd_config")
}

if ($sd_checkpoint) {
  [void]$ext_args.Add("--sd_checkpoint=$sd_checkpoint")
}

# run train
python main.py $ext_args

Write-Output "Train finished"
Read-Host | Out-Null ;
