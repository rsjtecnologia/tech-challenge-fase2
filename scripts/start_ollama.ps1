param(
    [string]$Model = "llama3.2:latest",
    [string]$BaseImage = "docker.io/ollama/ollama:latest",
    [string]$Image = "tech-challenge-ollama:local",
    [string]$ContainerName = "tech-challenge-ollama"
)

$ErrorActionPreference = "Stop"

$podmanCommand = Get-Command podman -ErrorAction SilentlyContinue
if ($podmanCommand) {
    $Podman = $podmanCommand.Source
} else {
    $localPodman = Join-Path $env:LOCALAPPDATA "Programs\Podman\podman.exe"
    if (Test-Path $localPodman) {
        $Podman = $localPodman
    } else {
        throw "Podman nao encontrado. Instale o Podman Desktop/CLI e abra um novo terminal."
    }
}

function Invoke-Podman {
    param([string[]]$Arguments)
    & $Podman @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Comando Podman falhou (codigo $LASTEXITCODE): podman $($Arguments -join ' ')"
    }
}

$machineOutput = & $Podman machine list --format json
if ($LASTEXITCODE -ne 0) {
    throw "Nao foi possivel consultar as maquinas Podman. Verifique a instalacao do Podman."
}

$machines = @()
if ($machineOutput) {
    $parsedMachines = ConvertFrom-Json -InputObject ($machineOutput -join "`n")
    if ($null -ne $parsedMachines) {
        $machines = @($parsedMachines | Where-Object {
            $_.Name -and -not [string]::IsNullOrWhiteSpace([string]$_.Name)
        })
    }
}

if ($machines.Count -eq 0) {
    $hostComputer = Get-CimInstance Win32_ComputerSystem
    if (-not $hostComputer.HypervisorPresent) {
        throw "O hipervisor nao esta ativo. Habilite virtualizacao Intel VT-x/AMD-V no firmware, WSL 2 e Plataforma de Maquina Virtual em PowerShell como Administrador (`wsl --install --no-distribution`); reinicie o Windows e execute novamente."
    }

    Write-Host "Criando a maquina Podman (4 CPUs, 6 GiB RAM, 30 GiB de disco)..."
    try {
        Invoke-Podman @("machine", "init", "--cpus", "4", "--memory", "6144", "--disk-size", "30", "--now")
    } catch {
        throw "Nao foi possivel criar a maquina Podman. Verifique WSL 2, Plataforma de Maquina Virtual e virtualizacao Intel VT-x/AMD-V; reinicie o Windows. Detalhe: $($_.Exception.Message)"
    }
} else {
    $runningMachines = @($machines | Where-Object { $_.Running })
    if ($runningMachines.Count -eq 0) {
        Write-Host "Iniciando maquina Podman: $($machines[0].Name)..."
        Invoke-Podman @("machine", "start", $machines[0].Name)
    }
}

Write-Host "Construindo imagem local $Image a partir de $BaseImage..."
$repoRoot = Split-Path -Parent $PSScriptRoot
$containerContext = Join-Path $repoRoot "infra\ollama"
$containerFile = Join-Path $containerContext "Containerfile"
Invoke-Podman @(
    "build",
    "--tag", $Image,
    "--build-arg", "OLLAMA_BASE_IMAGE=$BaseImage",
    "--file", $containerFile,
    $containerContext
)

$existingContainers = @(& $Podman ps -a --filter "name=^$ContainerName$" --format "{{.Names}}")
if ($LASTEXITCODE -ne 0) {
    throw "Nao foi possivel listar os containers Podman."
}

if ($existingContainers -contains $ContainerName) {
    $isRunning = & $Podman inspect --format "{{.State.Running}}" $ContainerName
    if ($LASTEXITCODE -ne 0) {
        throw "Nao foi possivel inspecionar o container Ollama existente."
    }
    if ($isRunning -ne "true") {
        Invoke-Podman @("start", $ContainerName)
    }
} else {
    Invoke-Podman @(
        "run", "-d",
        "--name", $ContainerName,
        "--publish", "127.0.0.1:11434:11434",
        "--volume", "tech-challenge-ollama-data:/root/.ollama",
        "--restart", "unless-stopped",
        $Image
    )
}

$ready = $false
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    try {
        $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3
        $ready = $true
        break
    } catch {
        Start-Sleep -Seconds 2
    }
}

if (-not $ready) {
    throw "O container Ollama iniciou, mas http://127.0.0.1:11434 nao respondeu. Confira `podman logs ollama` e o encaminhamento de portas do Podman."
}

Write-Host "Baixando modelo $Model (os arquivos ficam no volume persistente tech-challenge-ollama-data)..."
Invoke-Podman @("exec", $ContainerName, "ollama", "pull", $Model)

Write-Host "Pre-carregando o modelo na memoria (a primeira carga pode demorar)..."
$warmupBody = @{
    model = $Model
    messages = @()
    stream = $false
    keep_alive = "30m"
} | ConvertTo-Json -Depth 4
$null = Invoke-RestMethod -Uri "http://localhost:11434/api/chat" `
    -Method Post -ContentType "application/json" -Body $warmupBody -TimeoutSec 180

$env:LLM_PROVIDER = "ollama"
$env:LLM_MODEL = $Model
$env:LLM_BASE_URL = "http://localhost:11434"
if (-not $env:LLM_TIMEOUT_SECONDS) {
    $env:LLM_TIMEOUT_SECONDS = "180"
}

Write-Host ""
Write-Host "Ollama local pronto em http://127.0.0.1:11434"
Write-Host "Configuracao LLM_PROVIDER/LLM_MODEL definida neste terminal."
Write-Host "  .\run.ps1 -Provider Ollama"
