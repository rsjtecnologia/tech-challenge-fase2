# Prepara o ambiente: ativa o .venv se necessario e instala requirements.txt.
# Usa a versao mais recente do Python 3 instalada. Para manter o venv ativo na sessao atual, execute com:
#   . .\pre_execucao.ps1

$ErrorActionPreference = "Stop"

$raiz = $PSScriptRoot
if (-not $raiz) {
    $raiz = Get-Location
}

$venvPath = Join-Path $raiz ".venv"
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
$venvPython = Join-Path $venvPath "Scripts\python.exe"
$requirements = Join-Path $raiz "requirements.txt"

function Get-PythonMinorVersion {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @()
    )

    try {
        $output = & $Command @Arguments -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
    }
    catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0 -or -not $output) {
        return $null
    }
    return ($output | Select-Object -Last 1).Trim()
}

function Resolve-LatestPython {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $versoes = py -0p 2>$null |
            ForEach-Object {
                if ($_ -match '-V:(\d+\.\d+)') {
                    [version]$Matches[1]
                }
            } |
            Sort-Object -Descending -Unique

        if ($versoes) {
            $latest = $versoes[0]
            $minor = "$($latest.Major).$($latest.Minor)"
            $ver = Get-PythonMinorVersion -Command "py" -Arguments @("-$minor")
            if ($ver -eq $minor) {
                return @{ Command = "py"; Arguments = @("-$minor"); Version = $minor }
            }
        }

        $ver = Get-PythonMinorVersion -Command "py" -Arguments @("-3")
        if ($ver) {
            return @{ Command = "py"; Arguments = @("-3"); Version = $ver }
        }
    }

    foreach ($cmd in @("python3", "python")) {
        if (Get-Command $cmd -ErrorAction SilentlyContinue) {
            $ver = Get-PythonMinorVersion -Command $cmd
            if ($ver) {
                return @{ Command = $cmd; Arguments = @(); Version = $ver }
            }
        }
    }

    throw "Nenhum Python 3 encontrado. Instale o Python e garanta 'py' ou 'python' no PATH."
}

function Invoke-Python {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Python,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    & $Python.Command @($Python.Arguments + $Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao executar Python $($Python.Version) ($($Python.Command) $($Python.Arguments -join ' '))."
    }
}

$python = Resolve-LatestPython
$pythonVersao = $python.Version
Write-Host "Python detectado: $pythonVersao ($($python.Command) $($python.Arguments -join ' '))"

$venvEsperado = $null
if (Test-Path $venvPath) {
    $venvEsperado = (Resolve-Path $venvPath).Path
}

$venvAtivo = $env:VIRTUAL_ENV
$jaAtivoNesteProjeto = $venvAtivo -and $venvEsperado -and (
    [string]::Equals($venvAtivo, $venvEsperado, [System.StringComparison]::OrdinalIgnoreCase)
)

if (Test-Path $venvPython) {
    $venvVer = Get-PythonMinorVersion -Command $venvPython
    if (-not $venvVer -or $venvVer -ne $pythonVersao) {
        if ($jaAtivoNesteProjeto) {
            $motivo = if ($venvVer) { "Python $venvVer" } else { "venv invalido ou corrompido" }
            throw "O .venv ativo esta com $motivo. Desative o venv, apague a pasta .venv e execute o script de novo para recriar com Python $pythonVersao."
        }

        if (-not $venvVer) {
            Write-Host "venv existente invalido ou corrompido. Recriando com Python $pythonVersao ..."
        }
        else {
            Write-Host "venv existente usa Python $venvVer. Recriando com Python $pythonVersao ..."
        }
        cmd /c "rmdir /s /q `"$venvPath`"" 2>$null | Out-Null
        if (Test-Path $venvPath) {
            Remove-Item -LiteralPath $venvPath -Recurse -Force -ErrorAction SilentlyContinue
        }
        if (Test-Path $venvPath) {
            throw "Nao foi possivel remover $venvPath. Apague a pasta .venv manualmente e execute o script de novo."
        }
        $jaAtivoNesteProjeto = $false
        $venvEsperado = $null
    }
}

if ($jaAtivoNesteProjeto) {
    Write-Host "venv ja esta ativo: $venvAtivo"
}
else {
    if (-not (Test-Path $activateScript)) {
        Write-Host "venv nao encontrado. Criando .venv com Python $pythonVersao em $venvPath ..."
        Invoke-Python -Python $python -Arguments @("-m", "venv", $venvPath)
        if (-not (Test-Path $activateScript)) {
            throw "Falha ao criar o venv. Verifique se o Python $pythonVersao esta instalado."
        }
    }

    Write-Host "Ativando venv: $venvPath"
    . $activateScript
}

$pythonAtivoVer = Get-PythonMinorVersion -Command "python"
if ($pythonAtivoVer -ne $pythonVersao) {
    throw "O interpretador ativo e Python $pythonAtivoVer, mas este projeto exige $pythonVersao."
}

if (-not (Test-Path $requirements)) {
    throw "Arquivo nao encontrado: $requirements"
}

Write-Host "Instalando pacotes de requirements.txt (Python $pythonVersao) ..."
& $venvPython -m pip install -r $requirements
if ($LASTEXITCODE -ne 0) {
    throw "Falha ao instalar dependencias de requirements.txt. Verifique a saida do pip acima."
}

Write-Host "Ambiente pronto (Python $pythonVersao)."
