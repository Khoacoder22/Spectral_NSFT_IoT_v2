[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('plan', 'tune', 'evaluate', 'report', 'compare')]
    [string]$Mode,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PipelineArgs = @()
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$pipeline = Join-Path $projectRoot 'code\nnfst_competitor16_pipeline.py'
$comparison = Join-Path $projectRoot 'code\compare_nnfst_17models.py'

if (-not (Test-Path -LiteralPath $pipeline -PathType Leaf)) {
    throw "Could not find nNFST pipeline: $pipeline"
}

Push-Location -LiteralPath $projectRoot
try {
    if ($Mode -eq 'compare') {
        if (-not (Test-Path -LiteralPath $comparison -PathType Leaf)) {
            throw "Could not find comparison script: $comparison"
        }
        & python -u $comparison @PipelineArgs
    }
    else {
        & python -u $pipeline $Mode @PipelineArgs
    }

    if ($LASTEXITCODE -ne 0) {
        throw "nNFST command '$Mode' failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
