param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Context,

    [ValidateRange(300, 1800)]
    [int]$TimeoutSeconds = 600
)

$ErrorActionPreference = "Stop"
$manifest = Join-Path (Split-Path -Parent $PSScriptRoot) "infra\autoscaling\load-test-job.json"

function Invoke-Kubectl {
    param([string[]]$Arguments)
    $output = & kubectl --context $Context @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "kubectl falhou: kubectl $($Arguments -join ' ')"
    }
    return $output
}

function Get-ScaleState {
    $deployment = (Invoke-Kubectl @("-n", "tech-challenge", "get", "deployment", "diagnostics-api", "-o", "json") | Out-String) | ConvertFrom-Json
    $hpa = (Invoke-Kubectl @("-n", "tech-challenge", "get", "hpa", "diagnostics-api", "-o", "json") | Out-String) | ConvertFrom-Json
    $ready = [int]$deployment.status.readyReplicas
    Write-Host "$(Get-Date -Format o) desired=$($deployment.spec.replicas) ready=$ready"
    return @{ Desired = [int]$deployment.spec.replicas; Ready = $ready; Hpa = $hpa }
}

Get-Command kubectl -ErrorAction Stop | Out-Null
Invoke-Kubectl @("-n", "tech-challenge", "rollout", "status", "deployment/diagnostics-api", "--timeout=180s")
Invoke-Kubectl @("-n", "tech-challenge", "top", "pods", "-l", "app=diagnostics-api")
$initial = Get-ScaleState
$minimum = [int]$initial.Hpa.spec.minReplicas
if ($initial.Desired -ne $minimum -or $initial.Ready -ne $minimum) {
    throw "Aguarde a API estabilizar no minimo de $minimum replica(s), sem outra carga."
}
Invoke-Kubectl @("-n", "tech-challenge", "delete", "job", "diagnostics-load-test", "--ignore-not-found", "--wait=true")
$created = $false
try {
    Invoke-Kubectl @("apply", "-f", $manifest)
    $created = $true
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $scaledOut = $false
    $loadComplete = $false
    do {
        Start-Sleep -Seconds 10
        $state = Get-ScaleState
        if ($state.Ready -gt $minimum) { $scaledOut = $true }
        $job = (Invoke-Kubectl @("-n", "tech-challenge", "get", "job", "diagnostics-load-test", "-o", "json") | Out-String) | ConvertFrom-Json
        if ($job.status.failed -gt 0) {
            Invoke-Kubectl @("-n", "tech-challenge", "logs", "job/diagnostics-load-test")
            throw "O teste de carga falhou."
        }
        $loadComplete = $job.status.succeeded -eq 1
    } while (-not $loadComplete -and (Get-Date) -lt $deadline)
    if (-not $loadComplete) { throw "Timeout aguardando a conclusao da carga." }
    Invoke-Kubectl @("-n", "tech-challenge", "logs", "job/diagnostics-load-test")
    if (-not $scaledOut) { throw "Nao foi observado aumento de replicas prontas durante a carga." }
    do {
        Start-Sleep -Seconds 10
        $state = Get-ScaleState
    } while (($state.Desired -ne $minimum -or $state.Ready -ne $minimum) -and (Get-Date) -lt $deadline)
    if ($state.Desired -ne $minimum -or $state.Ready -ne $minimum) {
        throw "Nao foi observada reducao ao minimo dentro do timeout."
    }
    Write-Host "PASS: scale-out durante carga e scale-in apos carga, com requisicoes HTTP 200." -ForegroundColor Green
}
finally {
    if ($created) {
        Invoke-Kubectl @("-n", "tech-challenge", "delete", "job", "diagnostics-load-test", "--ignore-not-found", "--wait=true")
    }
}
