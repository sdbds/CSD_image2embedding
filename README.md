# CSD_image2embedding

code reference:https://huggingface.co/yuxi-liu-wired/CSD

## Installation

### Windows:

Clone this repo

`git clone https://github.com/sdbds/CSD_image2embedding` or Download in release

Powershell run with `Step1_install-uv.ps1`(right clik)

Then auto install(including python)

The install script also installs the default KMeans backend from [flash-kmeans](https://github.com/svg-project/flash-kmeans).

### Linux:
First use `bash Step0_for_linux_install_pwsh.bash`

Then `pwsh Step1_install-uv.ps1` or `sudo pwsh Step1_install-uv.ps1`

## Useage

1、Put any image datasets to `datasets` folder

2、Powershell run with `Step2_embedding.ps1`(right clik)
Linux use `pwsh Step2_embedding.ps1` or `sudo pwsh Step1_install-uv.ps1`

3、Open address in terminal(should be automatic)

4、results save in `output` folder

## Clustering backend

KMeans now defaults to `flash-kmeans`, with automatic fallback to the existing `scikit-learn` implementation if the accelerated backend is unavailable at runtime.
