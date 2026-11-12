[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet(
        "KNFST", "HHH", "HHHv2", "SpectralNFST",
        "KNN", "LDA", "NuSVC", "SGD", "GauNB", "NC", "RNC",
        "LGBM", "MLP", "TabNet", "FTTransformer", "SAINT"
    )]
    [string]$Model,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "../..")).Path

Push-Location $projectRoot
try {
    # Keep the same parameters as the original competitor runner.
    $fixedKernel = "l05_exponential_kernel"
    $fixedScaler = "QuantileTransformer"
    $fixedPoly = -1
    $fixedSeeds = @(42, 43, 44)

    # ONLY the two missing dataset variants.
    $datasets = [ordered]@{
        "IoTID20" = 1000
        "ToN_IoT" = 2000
    }

    $resultsRoot = Join-Path $projectRoot "results/knfst_comparison"
    $modelsRoot = Join-Path $resultsRoot "models"
    $tempRoot = Join-Path $resultsRoot ".tmp"
    $backupRoot = Join-Path $resultsRoot "backups"
    $modelOutput = Join-Path $modelsRoot "${Model}.csv"
    $trainScript = Join-Path $projectRoot "code/train_evaluate.py"

    if (-not (Test-Path -LiteralPath $trainScript)) {
        throw "Cannot find train script: $trainScript"
    }

    if (-not $DryRun) {
        foreach ($directory in @($modelsRoot, $tempRoot, $backupRoot)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
    }

    $dependencyChecks = @{
        "LGBM" = @{ Code = "import lightgbm"; Install = "python -m pip install lightgbm" }
        "TabNet" = @{ Code = "import torch; import pytorch_tabnet"; Install = "python -m pip install torch pytorch-tabnet" }
        "FTTransformer" = @{ Code = "import torch; import tab_transformer_pytorch"; Install = "python -m pip install torch tab-transformer-pytorch" }
        "SAINT" = @{ Code = "import torch"; Install = "python -m pip install torch" }
    }

    if (-not $DryRun -and $dependencyChecks.ContainsKey($Model)) {
        $check = $dependencyChecks[$Model]
        $savedPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            & python -c $check.Code 2>$null
            $dependencyExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $savedPreference
        }
        if ($dependencyExitCode -ne 0) {
            throw "Missing dependency for $Model. Install: $($check.Install)"
        }
    }

    Write-Host "`nMODEL    : $Model" -ForegroundColor Cyan
    Write-Host "Datasets : $($datasets.Keys -join ', ')"
    Write-Host "Seeds    : $($fixedSeeds -join ', ')"
    Write-Host "Kernel   : $fixedKernel"
    Write-Host "Output   : $modelOutput"
    Write-Host "DryRun   : $DryRun"

    # One automatic backup per actual execution, outside the models/ folder.
    if (-not $DryRun -and (Test-Path -LiteralPath $modelOutput)) {
        $stamp = Get-Date -Format "yyyyMMdd_HHmmss_fff"
        $backupFile = Join-Path $backupRoot "${Model}_before_missing_${stamp}.csv"
        Copy-Item -LiteralPath $modelOutput -Destination $backupFile -ErrorAction Stop
        Write-Host "BACKUP   : $backupFile" -ForegroundColor DarkCyan
    }

    $skipped = 0
    $executed = 0
    $pending = 0

    foreach ($dataset in $datasets.Keys) {
        $limit = [int]$datasets[$dataset]
        $dataType = "${dataset}_${limit}"
        Write-Host "`n=== $dataType ===" -ForegroundColor Cyan

        foreach ($seed in $fixedSeeds) {
            $existingRows = if (Test-Path -LiteralPath $modelOutput) {
                @(Import-Csv -LiteralPath $modelOutput)
            }
            else {
                @()
            }

            $completed = @(
                $existingRows | Where-Object {
                    $_.'Data Type' -eq $dataType -and
                    "$($_.Seed)" -eq "$seed"
                }
            )

            if ($completed.Count -gt 1) {
                throw "Duplicate cached rows: model=$Model dataset=$dataType seed=$seed"
            }
            if ($completed.Count -eq 1) {
                Write-Host "[SKIP] $Model | $dataType | seed=$seed" -ForegroundColor DarkYellow
                $skipped++
                continue
            }

            if ($DryRun) {
                Write-Host "[TODO] $Model | $dataType | seed=$seed" -ForegroundColor Yellow
                $pending++
                continue
            }

            $tempFile = Join-Path $tempRoot "${Model}__${dataType}__seed${seed}.csv"
            if (Test-Path -LiteralPath $tempFile) {
                Remove-Item -LiteralPath $tempFile -Force
            }

            $arguments = @(
                "code/train_evaluate.py",
                "--dataset", $dataset,
                "--limit", $limit,
                "--model", $Model,
                "--kernel", $fixedKernel,
                "--scaler", $fixedScaler,
                "--poly", $fixedPoly,
                "--seed", $seed,
                "--output", $tempFile
            )

            Write-Host "[RUN ] $Model | $dataType | seed=$seed" -ForegroundColor Green
            & python @arguments
            $exitCode = $LASTEXITCODE
            if ($exitCode -ne 0) {
                throw "Training failed: model=$Model dataset=$dataType seed=$seed exit=$exitCode"
            }
            if (-not (Test-Path -LiteralPath $tempFile)) {
                throw "Missing result CSV: $tempFile"
            }

            $newRows = @(Import-Csv -LiteralPath $tempFile)
            if ($newRows.Count -ne 1) {
                throw "Expected exactly 1 result row, got $($newRows.Count): $tempFile"
            }
            if ($newRows[0].'Data Type' -ne $dataType) {
                throw "Dataset mismatch: expected $dataType, got $($newRows[0].'Data Type')"
            }

            # Same column layout as the original runner, avoiding duplicate headers.
            $ordered = [ordered]@{
                Seed = $seed
                "Input Model" = $Model
            }
            foreach ($property in $newRows[0].PSObject.Properties) {
                if ($property.Name -in @("Seed", "Input Model")) {
                    continue
                }
                $ordered[$property.Name] = $property.Value
            }
            $resultRow = [pscustomobject]$ordered

            if ($existingRows.Count -gt 0) {
                $existingHeaders = @($existingRows[0].PSObject.Properties.Name)
                $newHeaders = @($resultRow.PSObject.Properties.Name)
                $differences = @(Compare-Object -ReferenceObject $existingHeaders -DifferenceObject $newHeaders)
                if ($differences.Count -gt 0) {
                    throw "CSV column mismatch for $modelOutput. Original preserved. Temp: $tempFile"
                }
                $resultRow = $resultRow | Select-Object -Property $existingHeaders
                $resultRow | Export-Csv -LiteralPath $modelOutput -Append -NoTypeInformation -Encoding utf8
            }
            else {
                $resultRow | Export-Csv -LiteralPath $modelOutput -NoTypeInformation -Encoding utf8
            }

            $executed++
            Write-Host "[SAVE] $dataType | seed=$seed -> $modelOutput" -ForegroundColor Green
            Remove-Item -LiteralPath $tempFile -Force
        }
    }

    if (-not $DryRun -and (Test-Path -LiteralPath $modelOutput)) {
        $finalRows = @(Import-Csv -LiteralPath $modelOutput)
        $duplicates = @(
            $finalRows |
                Group-Object -Property 'Data Type', Seed |
                Where-Object { $_.Count -gt 1 }
        )
        if ($duplicates.Count -gt 0) {
            throw "Duplicate dataset-seed rows found in $modelOutput"
        }
        Write-Host "`nDONE: $Model has $($finalRows.Count) rows in its CSV." -ForegroundColor Green
    }

    Write-Host "Skipped existing : $skipped"
    Write-Host "Executed now     : $executed"
    Write-Host "Pending (dry-run): $pending"
    Write-Host "Model CSV        : $modelOutput" -ForegroundColor Cyan
}
finally {
    Pop-Location
}
