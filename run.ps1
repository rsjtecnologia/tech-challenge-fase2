param(
    [switch]$NoBrowser,

    [ValidateSet("Gemini", "OpenAI", "Groq", "Qwen", "Ollama")]
    [string]$Provider = "Gemini",

    [ValidateNotNullOrEmpty()]
    [string]$Model,

    [ValidateRange(1, 65535)]
    [int]$Port = 8020,

    [ValidateRange(1, 300)]
    [int]$TimeoutSeconds = 45,

    [ValidateSet("ap-southeast-1", "cn-beijing", "eu-central-1", "cn-hongkong", "ap-northeast-1", "us-east-1")]
    [string]$QwenRegion = "ap-southeast-1"
)

$ErrorActionPreference = "Stop"
if ($Provider -ne "Qwen" -and $PSBoundParameters.ContainsKey("QwenRegion")) {
    throw "-QwenRegion so pode ser usado com -Provider Qwen."
}

$root = $PSScriptRoot
$application = Join-Path $root "src\web\app.py"
$preExecution = Join-Path $root "pre_execucao.ps1"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $application -PathType Leaf)) {
    throw "Aplicacao web nao encontrada: $application"
}
if (-not (Test-Path -LiteralPath $preExecution -PathType Leaf)) {
    throw "Script de preparacao nao encontrado: $preExecution"
}

$listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
if (@($listeners | Where-Object { $_.Port -eq $Port }).Count -gt 0) {
    throw "A porta $Port da aplicacao esta ocupada. Encerre a execucao anterior com Ctrl+C ou escolha outra porta, por exemplo .\run.ps1 -Port $($Port % 65535 + 1). Nenhum processo foi encerrado."
}

# Keep the active virtual environment in this process for the server lifetime.
. $preExecution
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Python do ambiente virtual nao encontrado: $venvPython."
}
Set-Location $root

$environmentNames = @(
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT_SECONDS",
    "LLM_NUM_PREDICT",
    "LLM_ENABLE_THINKING",
    "LLM_REASONING_EFFORT",
    "APP_PROFILE"
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
$key = $null
$secureKey = $null
$pointer = [IntPtr]::Zero
$browserProcess = $null

try {
    if ($Provider -eq "Ollama") {
        if (-not $Model) { $Model = "llama3.2:latest" }
        $ollamaScript = Join-Path $root "scripts\start_ollama.ps1"
        if (-not (Test-Path -LiteralPath $ollamaScript -PathType Leaf)) {
            throw "Preparador do Ollama nao encontrado: $ollamaScript"
        }
        Write-Host "Preparando Podman/Ollama local..." -ForegroundColor Cyan
        & $ollamaScript -Model $Model
        if ($env:LLM_PROVIDER -ne "ollama" -or $env:LLM_MODEL -ne $Model) {
            throw "Ollama nao concluiu a configuracao. A aplicacao nao sera iniciada."
        }
        $env:APP_PROFILE = "full"
        if (-not $env:LLM_TIMEOUT_SECONDS) { $env:LLM_TIMEOUT_SECONDS = "180" }
    }
    else {
        if ($Provider -eq "Gemini") {
            $baseUrl = "https://generativelanguage.googleapis.com/v1beta/openai"
            $key = $env:GEMINI_API_KEY
            if (-not $Model) { $Model = "gemini-3.1-flash-lite" }
            Write-Host "Crie sua chave em https://aistudio.google.com/apikey e confira a cota do modelo."
            Write-Host "No Free Tier, o conteudo pode ser usado para melhorar produtos Google. Use somente dados agregados/simulados."
        }
        elseif ($Provider -eq "Groq") {
            $baseUrl = "https://api.groq.com/openai/v1"
            $key = $env:GROQ_API_KEY
            if (-not $Model) { $Model = "llama-3.1-8b-instant" }
        }
        elseif ($Provider -eq "Qwen") {
            $baseUrl = "https://$QwenRegion.maas.qwencloudapi.com/compatible-mode/v1"
            $key = $env:DASHSCOPE_API_KEY
            if (-not $Model) { $Model = "qwen-flash" }
            Write-Host "QwenCloud: regiao $QwenRegion; chave API pay-as-you-go."
            Write-Host "Crie a chave em https://home.qwencloud.com/api-keys."
        }
        else {
            $baseUrl = "https://api.openai.com/v1"
            $key = $env:OPENAI_API_KEY
            if (-not $Model) { $Model = "gpt-4o-mini" }
        }
        if ([string]::IsNullOrWhiteSpace($Model) -or $Model -match "\s") {
            throw "Informe um identificador de modelo valido para $Provider."
        }

        Write-Host "Modo online: $Provider / $Model. Sujeito a cobranca e limites da conta." -ForegroundColor Cyan
        Write-Host "Somente contexto agregado/simulado e a pergunta sao enviados ao provedor. Nao informe dados identificaveis."
        if ([string]::IsNullOrWhiteSpace($key)) {
            $secureKey = Read-Host "Chave API de $Provider (entrada oculta; nao sera salva)" -AsSecureString
            $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
            try {
                $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
            }
            finally {
                [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
                $pointer = [IntPtr]::Zero
                $secureKey.Dispose()
                $secureKey = $null
            }
        }
        if ([string]::IsNullOrWhiteSpace($key)) {
            throw "Chave API ausente. A aplicacao nao sera iniciada."
        }
        $key = $key.Trim()

        $env:LLM_PROVIDER = "openai_compatible"
        $env:LLM_BASE_URL = $baseUrl
        $env:LLM_MODEL = $Model
        $env:LLM_API_KEY = $key
        $env:LLM_TIMEOUT_SECONDS = [string]$TimeoutSeconds
        $env:LLM_NUM_PREDICT = $null
        $env:LLM_ENABLE_THINKING = if ($Provider -eq "Qwen") { "false" } else { $null }
        $env:LLM_REASONING_EFFORT = if ($Provider -eq "Gemini") { "minimal" } else { $null }
        $env:APP_PROFILE = "full"
    }

    Write-Host "Verificando dados locais e geracao do provedor..." -ForegroundColor Cyan
    $previousErrorPreference = $ErrorActionPreference
    try {
        # Capture native stderr without PowerShell replacing it with NativeCommandError.
        $ErrorActionPreference = "Continue"
        $preflightOutput = & $venvPython -m src.web.preflight 2>&1
        $preflightExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previousErrorPreference
    }
    $preflightDetails = ($preflightOutput | ForEach-Object { "$_" }) -join [Environment]::NewLine
    if ($preflightExitCode -eq 2) {
        Write-Warning "$preflightDetails`nAPI/web iniciara com LLM nao validada. Consultas ao provedor podem falhar; nao houve troca de modelo nem simulacao de resposta."
    }
    elseif ($preflightExitCode -ne 0) {
        throw "Verificacao inicial falhou (codigo $preflightExitCode). Aplicacao nao iniciada.`n$preflightDetails"
    }
    else {
        Write-Host $preflightDetails
    }

    Write-Host ""
    Write-Host "API/web iniciando em http://127.0.0.1:$Port" -ForegroundColor Green
    Write-Host "Documentacao da API: http://127.0.0.1:$Port/docs"
    Write-Host "Mantenha este terminal aberto; Ctrl+C encerra o servidor."
    Write-Host ""
    $browserScript = Join-Path $root "scripts\open_browser.py"
    if (-not (Test-Path -LiteralPath $browserScript -PathType Leaf)) {
        throw "Verificador da API/navegador nao encontrado: $browserScript"
    }
    $browserArguments = @("`"$browserScript`"", "--port", [string]$Port)
    if ($NoBrowser) { $browserArguments += "--no-browser" }
    $browserProcess = Start-Process -FilePath $venvPython -ArgumentList $browserArguments -NoNewWindow -PassThru
    & $venvPython -m uvicorn src.web.app:app --host 127.0.0.1 --port $Port --reload
    if ($LASTEXITCODE -ne 0) {
        throw "A aplicacao API/web encerrou com codigo $LASTEXITCODE."
    }
}
finally {
    if ($browserProcess -and -not $browserProcess.HasExited) {
        Stop-Process -Id $browserProcess.Id -ErrorAction SilentlyContinue
    }
    if ($pointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
    if ($secureKey) { $secureKey.Dispose() }
    $key = $null
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], "Process")
    }
}
