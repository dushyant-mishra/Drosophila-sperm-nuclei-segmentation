param(
    [Parameter(Mandatory = $true)]
    [string]$ClaimId,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[A-Za-z0-9._-]+$')]
    [string]$RunId,

    [string[]]$Roles = @(),
    [switch]$Parallel,
    [switch]$AllowDirty
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$RegistryPath = Join-Path $RepoRoot 'audits\claims_registry.json'
$SchemaPath = Join-Path $RepoRoot 'audits\review_schema.json'
$DecisionContextPath = Join-Path $RepoRoot 'audits\V5_7_1_DESIGN_DECISIONS.md'
$RunRoot = Join-Path $RepoRoot ("audits\runs\{0}" -f $RunId)
$ReviewRoot = Join-Path $RunRoot 'reviews'
$PromptRoot = Join-Path $RunRoot 'prompts'

$registry = Get-Content -LiteralPath $RegistryPath -Raw | ConvertFrom-Json
$claim = @($registry.claims | Where-Object claim_id -eq $ClaimId)
if ($claim.Count -ne 1) {
    throw "Expected exactly one claim named '$ClaimId'; found $($claim.Count)."
}
$claim = $claim[0]

if ($Roles.Count -eq 0) {
    $Roles = @($claim.required_roles)
}
$Roles = @($Roles | Select-Object -Unique)
foreach ($role in $Roles) {
    $charter = Join-Path $RepoRoot ("audits\agent_charters\{0}.md" -f $role)
    if (-not (Test-Path -LiteralPath $charter)) {
        throw "Unknown role '$role'; charter not found: $charter"
    }
}

$commit = (& git -C $RepoRoot rev-parse HEAD).Trim()
$branchOutput = & git -C $RepoRoot branch --show-current
$branch = if ([string]::IsNullOrWhiteSpace([string]$branchOutput)) {
    'DETACHED'
} else {
    ([string]$branchOutput).Trim()
}
$statusLines = @(& git -C $RepoRoot status --porcelain=v1)
$dirty = $statusLines.Count -gt 0
if ($dirty -and -not $AllowDirty) {
    throw 'Working tree is not clean. Commit/stash changes or rerun with -AllowDirty for a pre-commit audit.'
}

New-Item -ItemType Directory -Force -Path $ReviewRoot, $PromptRoot | Out-Null
$claimSnapshot = Join-Path $RunRoot 'claim_snapshot.json'
$claim | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $claimSnapshot -Encoding utf8

$manifest = [ordered]@{
    schema_version = '1.0'
    audit_run_id = $RunId
    claim_id = $ClaimId
    reviewed_commit = $commit
    branch = $branch
    dirty_worktree = $dirty
    working_tree_status = $statusLines
    mode = if ($dirty) { 'pre_commit' } else { 'acceptance_candidate' }
    roles = $Roles
    started_at_utc = [DateTime]::UtcNow.ToString('o')
    completed_at_utc = $null
    role_exit_codes = [ordered]@{}
}
$manifestPath = Join-Path $RunRoot 'manifest.json'
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8

$codexCommand = (Get-Command codex -ErrorAction Stop).Source
$decisionContext = if (Test-Path -LiteralPath $DecisionContextPath) {
    Get-Content -LiteralPath $DecisionContextPath -Raw
} else {
    'No project decision ledger was supplied.'
}
$tasks = @()
foreach ($role in $Roles) {
    $charterPath = Join-Path $RepoRoot ("audits\agent_charters\{0}.md" -f $role)
    $promptPath = Join-Path $PromptRoot ("{0}.md" -f $role)
    $outputPath = Join-Path $ReviewRoot ("{0}.json" -f $role)
    $charterText = Get-Content -LiteralPath $charterPath -Raw
    $claimText = $claim | ConvertTo-Json -Depth 20
    $prompt = @"
You are the independent Saturn reviewer for role: $role.

Audit run: $RunId
Claim: $ClaimId
Reviewed Git commit: $commit
Working tree mode: $($manifest.mode)

Read the repository and evaluate the claim. Do not edit files, do not run a
full image stack, do not commit, and do not push. You may run focused read-only
inspection and tests. Do not read other reviewers' outputs. Treat missing
evidence as missing; do not infer that another agent checked it.

Use $RepoRoot\.venv\Scripts\python.exe for Python tests; system Python is not
the validated project environment. Prefer the commit-bound evidence and
validation receipt under audits over rerunning expensive image inference.

$charterText

CLAIM SNAPSHOT:
$claimText

PROJECT DECISION CONTEXT:
$decisionContext

Treat the decision context as intended behavior to verify, not as proof that
the implementation is correct. Report any mismatch between intent and code.

Your final response must be JSON matching audits/review_schema.json. Use the
exact audit_run_id, claim_id, role, and reviewed_commit above. Every pass/fail
check and every finding must cite concrete evidence such as path:line, a test
name, a command result, or a generated artifact path. A conditional verdict
does not pass the acceptance gate.
"@
    Set-Content -LiteralPath $promptPath -Value $prompt -Encoding utf8
    $tasks += @{
        Role = $role
        PromptPath = $promptPath
        OutputPath = $outputPath
        TranscriptPath = Join-Path $RunRoot ("logs\{0}.log" -f $role)
    }
}

function Invoke-AuditRole {
    param($Task, $Codex, $Root, $Schema)
    New-Item -ItemType Directory -Force -Path (Split-Path $Task['TranscriptPath']) | Out-Null
    $promptText = Get-Content -LiteralPath $Task['PromptPath'] -Raw
    $arguments = @(
        'exec', '--sandbox', 'read-only', '--ephemeral', '--cd', $Root,
        '--output-schema', $Schema, '--output-last-message', $Task['OutputPath'],
        '-'
    )
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        # Windows PowerShell converts native stderr lines into ErrorRecord
        # objects. Codex may emit non-fatal warnings there, so preserve them in
        # the transcript and use the native exit code as the source of truth.
        $ErrorActionPreference = 'Continue'
        $promptText | & $Codex @arguments *> $Task['TranscriptPath']
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    return $exitCode
}

if ($Parallel) {
    $jobs = foreach ($task in $tasks) {
        Start-Job -ScriptBlock ${function:Invoke-AuditRole} -ArgumentList @(
            $task, $codexCommand, $RepoRoot, $SchemaPath
        )
    }
    Wait-Job -Job $jobs | Out-Null
    for ($index = 0; $index -lt $jobs.Count; $index++) {
        $result = @(Receive-Job -Job $jobs[$index] -ErrorAction SilentlyContinue `
            -WarningAction SilentlyContinue)
        $exitCode = if ($result.Count) {
            [int]$result[-1]
        } elseif (Test-Path -LiteralPath $tasks[$index]['OutputPath']) {
            0
        } else {
            1
        }
        $manifest.role_exit_codes[$tasks[$index]['Role']] = $exitCode
        Remove-Job -Job $jobs[$index]
    }
} else {
    foreach ($task in $tasks) {
        Write-Host "`n=== Independent review: $($task['Role']) ==="
        $manifest.role_exit_codes[$task['Role']] = Invoke-AuditRole `
            -Task $task -Codex $codexCommand -Root $RepoRoot -Schema $SchemaPath
    }
}

$manifest.completed_at_utc = [DateTime]::UtcNow.ToString('o')
$manifest | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $manifestPath -Encoding utf8

& (Join-Path $RepoRoot '.venv\Scripts\python.exe') `
    (Join-Path $RepoRoot 'scripts\validate_agent_audit.py') --run $RunRoot
exit $LASTEXITCODE
