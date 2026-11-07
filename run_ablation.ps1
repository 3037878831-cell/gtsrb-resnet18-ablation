param(
    [string[]]$Variants = @("cbam", "multiscale", "gelu", "mish", "groupnorm")
)

$ErrorActionPreference = "Stop"
$Python = "$PSScriptRoot\..\.conda\gtsrb\python.exe"
foreach ($Variant in $Variants) {
    Write-Host "Starting experiment: $Variant"
    & $Python "$PSScriptRoot\train.py" --variant $Variant --epochs 30 `
        --batch-size 128 --workers 4 --lr 0.001 --early-stopping 8
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $Variant"
    }
}

