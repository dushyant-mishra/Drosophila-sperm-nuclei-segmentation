param(
    [int]$SegmentationCandidateCount = 12,
    [int]$UnetCandidateCount = 12,
    [int]$Seed = 12345,
    [string]$SegmentationCandidateRole = "reviewed_base",
    [string]$UnetCandidateRole = "evidence_0.05_0.30",
    [string]$RunId = "",
    [switch]$UnetOnly,
    [string]$ClassicalRunId = "",
    [string]$UnetCacheRunId = "",
    [switch]$Resume,
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Tuner = Join-Path $ProjectRoot "utils\tune_parameters_Saturnv5_7.py"
$ManifestPath = Join-Path $PSScriptRoot "mixed_tuner_manifest.csv"
$ResultsRoot = Join-Path $PSScriptRoot "results"
$Checkpoint = Join-Path $ProjectRoot "Kaggle notebook outputs\v57_kj_wt_training_export\checkpoints\epoch_003.pt"
$StartingPreset = Join-Path $ProjectRoot "parameter_tuning_results_v5_7\epoch003_kj_wt_shared\shared_unet_rescue_params_v5_7_001.json"

foreach ($required in @(
    $Python,
    $Tuner,
    $ManifestPath,
    $Checkpoint,
    $StartingPreset
)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Required file not found: $required"
    }
}

$rows = @(Import-Csv -LiteralPath $ManifestPath)
if ($rows.Count -ne 4) {
    throw "Expected four mixed-tuner strata, found $($rows.Count)"
}

$groupCounts = $rows | Group-Object group
foreach ($group in @("WT", "KJ")) {
    $count = ($groupCounts | Where-Object Name -eq $group).Count
    if ($count -ne 2) {
        throw "Expected two $group strata, found $count"
    }
}

$validatedRows = @()
foreach ($row in $rows) {
    if (-not (Test-Path -LiteralPath $row.image_dir -PathType Container)) {
        throw "Image directory not found for $($row.specimen_id): $($row.image_dir)"
    }
    if (-not (Test-Path -LiteralPath $row.roi_path -PathType Leaf)) {
        throw "ROI not found for $($row.specimen_id): $($row.roi_path)"
    }

    $sourceImages = @(
        Get-ChildItem -LiteralPath $row.image_dir -File |
            Where-Object Name -Match "(?i)_z\d+_ch00\.tiff?$"
    )
    if ($sourceImages.Count -ne [int]$row.source_slice_count) {
        throw (
            "Unexpected source-image count for $($row.specimen_id): " +
            "expected $($row.source_slice_count), found $($sourceImages.Count)"
        )
    }
    $validatedRows += $row
    Write-Host (
        "Validated $($row.specimen_id) [$($row.group)]: " +
        "$($sourceImages.Count) slices; Z=$($row.selected_z_indices)"
    )
}

if ($ValidateOnly) {
    Write-Host ""
    Write-Host "Mixed WT/KJ staged tuner input validation passed."
    exit 0
}

if (-not $RunId) {
    $RunId = Get-Date -Format "yyyyMMdd_HHmmss"
}
if ($UnetOnly -and -not $ClassicalRunId) {
    throw "-UnetOnly requires -ClassicalRunId with a completed classical stage"
}
$RunRoot = Join-Path $ResultsRoot $RunId
if (Test-Path -LiteralPath $RunRoot) {
    if (-not $Resume) {
        throw (
            "Run directory already exists. Use -RunId $RunId -Resume to continue it, " +
            "or choose another -RunId: $RunRoot"
        )
    }
    $metadataPath = Join-Path $RunRoot "run_metadata.json"
    if (-not (Test-Path -LiteralPath $metadataPath -PathType Leaf)) {
        throw "Cannot resume because run_metadata.json is missing: $metadataPath"
    }
    $existingMetadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    $resumeChecks = [ordered]@{
        seed = $Seed
        segmentation_candidate_count = $SegmentationCandidateCount
        unet_candidate_count = $UnetCandidateCount
        segmentation_candidate_role = $SegmentationCandidateRole
        unet_candidate_role = $UnetCandidateRole
        unet_cache_run_id = $UnetCacheRunId
    }
    foreach ($key in $resumeChecks.Keys) {
        if ([string]$existingMetadata.$key -ne [string]$resumeChecks[$key]) {
            throw (
                "Resume setting mismatch for ${key}: existing=$($existingMetadata.$key), " +
                "requested=$($resumeChecks[$key])"
            )
        }
    }
    Write-Host "Resuming staged tuner run: $RunRoot"
} else {
    if ($Resume) {
        throw "Cannot resume because the run directory does not exist: $RunRoot"
    }
    New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

    $metadata = [ordered]@{
        run_id = $RunId
        created_at = (Get-Date).ToString("o")
        seed = $Seed
        segmentation_candidate_count = $SegmentationCandidateCount
        unet_candidate_count = $UnetCandidateCount
        segmentation_candidate_role = $SegmentationCandidateRole
        unet_candidate_role = $UnetCandidateRole
        checkpoint = $Checkpoint
        starting_preset = $StartingPreset
        manifest = $ManifestPath
        unet_only = [bool]$UnetOnly
        classical_source_run_id = $ClassicalRunId
        unet_cache_run_id = $UnetCacheRunId
    }
    $metadata | ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $RunRoot "run_metadata.json") -Encoding UTF8
}

function Invoke-StratumTuning {
    param(
        [string]$Mode,
        [object]$Row,
        [string]$BasePreset,
        [string]$OutputDirectory,
        [int]$CandidateCount,
        [string]$ReviewCandidateRole = "",
        [string]$UnetCacheDirectory = ""
    )

    $resultPath = Join-Path $OutputDirectory "tuning_results_saturnv5_7_$Mode.json"
    if (Test-Path -LiteralPath $resultPath -PathType Leaf) {
        Write-Host ""
        Write-Host "=== Reusing completed $Mode result: $($Row.specimen_id) ==="
        return $resultPath
    }

    $arguments = @(
        $Tuner,
        "--mode", $Mode,
        "--dir", $Row.image_dir,
        "--slices", $Row.selected_z_indices,
        "--roi-mask", $Row.roi_path,
        "--auto-calibration",
        "--base-params", $BasePreset,
        "--outdir", $OutputDirectory,
        "--maxiter", $CandidateCount,
        "--seed", $Seed,
        "--review-candidates", 6
    )
    if ($ReviewCandidateRole) {
        $arguments += @("--review-candidate-role", $ReviewCandidateRole)
    }
    if ($Mode -eq "unet_rescue") {
        $arguments += @("--unet-model", $Checkpoint)
        if (
            $UnetCacheDirectory -and
            (Test-Path -LiteralPath $UnetCacheDirectory -PathType Container)
        ) {
            $arguments += @("--unet-cache-dir", $UnetCacheDirectory)
        }
    }

    Write-Host ""
    Write-Host "=== $Mode tuning: $($Row.specimen_id) [$($Row.group)] ==="
    & $Python @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "$Mode tuner failed for $($Row.specimen_id)"
    }

    if (-not (Test-Path -LiteralPath $resultPath -PathType Leaf)) {
        throw "Missing $Mode result for $($Row.specimen_id): $resultPath"
    }
    return $resultPath
}

function Invoke-SharedAggregation {
    param(
        [string]$Mode,
        [string[]]$ResultPaths,
        [string]$BasePreset,
        [string]$SelectedRole,
        [string]$OutputDirectory
    )

    $arguments = @(
        $Tuner,
        "--mode", $Mode,
        "--base-params", $BasePreset,
        "--shared-candidate-role", $SelectedRole,
        "--outdir", $OutputDirectory
    )
    if ($Mode -eq "unet_rescue") {
        $arguments += @("--unet-model", $Checkpoint)
    }
    foreach ($resultPath in $ResultPaths) {
        $arguments += @("--aggregate-stratum-results", $resultPath)
    }

    Write-Host ""
    Write-Host "=== Aggregating shared $Mode candidate: $SelectedRole ==="
    & $Python @arguments | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Shared $Mode aggregation failed"
    }

    $preset = Get-ChildItem -LiteralPath $OutputDirectory -File |
        Where-Object Name -Like "shared_${Mode}_params_v5_7_*.json" |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
    if ($null -eq $preset) {
        throw "Shared $Mode preset was not created in $OutputDirectory"
    }
    return $preset.FullName
}

# Stage 1: isolate and tune classical Saturn 2D segmentation, or reuse it
# unchanged when only the corrected U-Net rescue lane needs retuning.
if ($UnetOnly) {
    $sourceRunRoot = Join-Path $ResultsRoot $ClassicalRunId
    if (-not (Test-Path -LiteralPath $sourceRunRoot -PathType Container)) {
        throw "Classical source run not found: $sourceRunRoot"
    }
    $shared2dPreset = Get-ChildItem `
        -LiteralPath (Join-Path $sourceRunRoot "01_classical_2d\shared_wt_kj") `
        -Filter "shared_segmentation_params_v5_7_*.json" `
        -File |
        Sort-Object LastWriteTime |
        Select-Object -Last 1 -ExpandProperty FullName
    if (-not $shared2dPreset) {
        throw "Shared classical preset not found in source run: $sourceRunRoot"
    }
    Write-Host "Reusing unchanged shared classical preset: $shared2dPreset"
} else {
    $segmentationRoot = Join-Path $RunRoot "01_classical_2d"
    $segmentationResults = @()
    foreach ($row in $validatedRows) {
        $stratumOut = Join-Path $segmentationRoot $row.specimen_id
        $segmentationResults += Invoke-StratumTuning `
            -Mode "segmentation" `
            -Row $row `
            -BasePreset $StartingPreset `
            -OutputDirectory $stratumOut `
            -CandidateCount $SegmentationCandidateCount `
            -ReviewCandidateRole $SegmentationCandidateRole
    }
    $shared2dDirectory = Join-Path $segmentationRoot "shared_wt_kj"
    $shared2dPreset = Invoke-SharedAggregation `
        -Mode "segmentation" `
        -ResultPaths $segmentationResults `
        -BasePreset $StartingPreset `
        -SelectedRole $SegmentationCandidateRole `
        -OutputDirectory $shared2dDirectory
}

# Stage 2: tune U-Net rescue against the unchanged shared 2D base.
$unetRoot = Join-Path $RunRoot "02_unet_rescue"
$unetResults = @()
foreach ($row in $validatedRows) {
    $stratumOut = Join-Path $unetRoot $row.specimen_id
    $cacheDirectory = ""
    $cacheSourceRunId = $UnetCacheRunId
    if (-not $cacheSourceRunId -and $UnetOnly) {
        $cacheSourceRunId = $ClassicalRunId
    }
    if ($cacheSourceRunId) {
        $cacheDirectory = Join-Path (
            Join-Path $ResultsRoot $cacheSourceRunId
        ) "02_unet_rescue\$($row.specimen_id)\unet_probability_cache"
        if (-not (Test-Path -LiteralPath $cacheDirectory -PathType Container)) {
            throw (
                "U-Net probability cache not found for $($row.specimen_id) " +
                "in run ${cacheSourceRunId}: $cacheDirectory"
            )
        }
    }
    $unetResults += Invoke-StratumTuning `
        -Mode "unet_rescue" `
        -Row $row `
        -BasePreset $shared2dPreset `
        -OutputDirectory $stratumOut `
        -CandidateCount $UnetCandidateCount `
        -ReviewCandidateRole $UnetCandidateRole `
        -UnetCacheDirectory $cacheDirectory
}
$sharedHybridDirectory = Join-Path $unetRoot "shared_wt_kj"
$sharedHybridPreset = Invoke-SharedAggregation `
    -Mode "unet_rescue" `
    -ResultPaths $unetResults `
    -BasePreset $shared2dPreset `
    -SelectedRole $UnetCandidateRole `
    -OutputDirectory $sharedHybridDirectory

$completion = [ordered]@{
    run_id = $RunId
    completed_at = (Get-Date).ToString("o")
    shared_2d_preset = $shared2dPreset
    shared_hybrid_preset = $sharedHybridPreset
    selection_status = "candidates_for_visual_inspection"
}
$completion | ConvertTo-Json -Depth 4 |
    Set-Content -LiteralPath (Join-Path $RunRoot "completed_run.json") -Encoding UTF8

Write-Host ""
Write-Host "Staged mixed WT/KJ tuning complete."
Write-Host "Run directory: $RunRoot"
Write-Host "Shared 2D candidate: $shared2dPreset"
Write-Host "Shared hybrid candidate: $sharedHybridPreset"
Write-Host "Review every stratum PDF before accepting either preset."
