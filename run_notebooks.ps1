# Executa notebooks do pipeline por estudo (dataset).
# Uso: .\run_notebooks.ps1

param(
    [ValidateRange(1, 65535)]
    [int]$ApiPort = 8020
)

$ErrorActionPreference = "Stop"

# Evita acentos quebrados no console Windows (cp1252).
try {
    $utf8 = [System.Text.UTF8Encoding]::new()
    [Console]::OutputEncoding = $utf8
    [Console]::InputEncoding = $utf8
    $OutputEncoding = $utf8
    chcp 65001 | Out-Null
}
catch {
    # Continua mesmo se o console nao permitir alterar o code page.
}

$raiz = $PSScriptRoot
$preExecucao = Join-Path $raiz "pre_execucao.ps1"
$executor = Join-Path $raiz "run_notebook.py"
$venvPython = Join-Path $raiz ".venv\Scripts\python.exe"
$notebooksInterativos = @{
    "1" = @{
        Nome = "Câncer de mama - Fase 2 (interpretação)"
        Caminho = "notebooks\breast-cancer-wisconsin-data\08_export_web.ipynb"
    }
    "2" = @{
        Nome = "PCOS - Fase 2 (interpretação)"
        Caminho = "notebooks\polycystic-ovary-syndrome-pcos\05_export_web.ipynb"
    }
    "3" = @{
        Nome = "Rotas médicas - Fase 2"
        Caminho = "notebooks\medical-routes\01_otimizacao_rotas.ipynb"
    }
}

if (-not (Test-Path $preExecucao)) {
    throw "Arquivo de preparação não encontrado: $preExecucao"
}

if (-not (Test-Path $executor)) {
    throw "Executor de notebook não encontrado: $executor"
}

# O dot sourcing mantém o ambiente virtual ativo neste processo.
. $preExecucao

if (-not (Test-Path $venvPython)) {
    throw "Python do venv nao encontrado: $venvPython. Execute pre_execucao.ps1 novamente."
}

Set-Location $raiz

$estudos = @{
    "1" = @{
        Nome = "Câncer de mama (Wisconsin)"
        Pasta = "notebooks/breast-cancer-wisconsin-data"
        Notebooks = @(
            "03_preprocessamento.ipynb"
            "04_modelagem_treino.ipynb"
            "05_avaliacao_metricas.ipynb"
            "06_explicabilidade.ipynb"
            "07_discussao_demo.ipynb"
            "08_export_web.ipynb"
        )
    }
    "2" = @{
        Nome = "SOP (PCOS)"
        Pasta = "notebooks/polycystic-ovary-syndrome-pcos"
        Notebooks = @(
            "02_preprocessamento.ipynb"
            "03_modelagem_treino.ipynb"
            "04_avaliacao_metricas.ipynb"
            "05_export_web.ipynb"
        )
    }
}

function Show-EstudoMenu {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Escolha o estudo"
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""
    foreach ($chave in ($estudos.Keys | Sort-Object)) {
        Write-Host ("  [{0}] {1}" -f $chave, $estudos[$chave].Nome)
    }
    Write-Host ""
    Write-Host "  [T] Executar todos os estudos (na ordem)"
    Write-Host "  [3] Executar otimização genética dos modelos (Fase 2)"
    Write-Host "  [4] Abrir notebooks interativos (Fase 2: modelos e rotas)"
    Write-Host "  [S] Sair"
    Write-Host ""
}

function Invoke-ModelOptimization {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("breast_cancer", "pcos", "all")]
        [string]$Study
    )

    $datasets = @{
        breast_cancer = Join-Path $raiz "src\db\models\breast-cancer-wisconsin-data\data.csv"
        pcos = Join-Path $raiz "src\db\models\polycystic-ovary-syndrome-pcos\PCOS_infertility.csv"
    }
    $requiredStudies = if ($Study -eq "all") {
        @("breast_cancer", "pcos")
    } else {
        @($Study)
    }
    foreach ($requiredStudy in $requiredStudies) {
        if (-not (Test-Path -LiteralPath $datasets[$requiredStudy] -PathType Leaf)) {
            throw "Dataset necessário para a otimização não encontrado: $($datasets[$requiredStudy])"
        }
    }

    Write-Host ""
    Write-Host "Executando os três experimentos genéticos por modelo. Esta etapa pode levar vários minutos." -ForegroundColor Yellow
    Write-Host "Resultados e logs serão salvos em src\db\outputs\phase2-genetic."
    Set-Location $raiz
    & $venvPython -m src.optimization.cli --study $Study
    if ($LASTEXITCODE -ne 0) {
        throw "Falha na otimização genética dos modelos (estudo: $Study)."
    }
    Write-Host "Otimização genética concluída: $Study" -ForegroundColor Green
}

function Invoke-ModelOptimizationMenu {
    while ($true) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Otimização genética de modelos - Fase 2"
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  [1] Câncer de mama"
        Write-Host "  [2] PCOS"
        Write-Host "  [T] Ambos os estudos"
        Write-Host "  [B] Voltar"
        Write-Host ""
        $choice = (Read-Host "Escolha uma opção").Trim()

        if ($choice -match '^[Bb]$') {
            return
        }
        $study = switch -Regex ($choice) {
            '^1$' { "breast_cancer"; break }
            '^2$' { "pcos"; break }
            '^[Tt]$' { "all"; break }
            default { $null }
        }
        if (-not $study) {
            Write-Host "Opção inválida." -ForegroundColor Red
            continue
        }
        try {
            Invoke-ModelOptimization -Study $study
        }
        catch {
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
        return
    }
}

function Start-InteractiveNotebook {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Notebook
    )

    $path = Join-Path $raiz $Notebook.Caminho
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Notebook interativo não encontrado: $path"
    }
    Write-Host ""
    Write-Host "Abrindo $($Notebook.Nome) no Jupyter..." -ForegroundColor Cyan
    if ($Notebook.Caminho -match "medical-routes") {
        Write-Host "Inicie .\run.ps1 em outro terminal antes de executar a célula da interface."
        Write-Host "A fonte Olist exige os CSVs em src\db\models\dataset_rotasBR; escolha demo para dados fictícios."
    }
    elseif ($Notebook.Caminho -match "08_export_web|05_export_web") {
        Write-Host "Para a seção interativa, execute .\run.ps1 em outro terminal e gere antes os resultados genéticos."
    }
    Set-Location $raiz
    $variableName = "TECH_CHALLENGE_API_BASE_URL"
    $previousApiUrl = [Environment]::GetEnvironmentVariable($variableName, "Process")
    try {
        [Environment]::SetEnvironmentVariable(
            $variableName,
            "http://127.0.0.1:$ApiPort",
            "Process"
        )
        & $venvPython -m notebook $path
        if ($LASTEXITCODE -ne 0) {
            throw "Não foi possível abrir o Jupyter. Verifique a instalação do pacote notebook."
        }
    }
    finally {
        [Environment]::SetEnvironmentVariable($variableName, $previousApiUrl, "Process")
    }
}

function Invoke-InteractiveNotebookMenu {
    while ($true) {
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  Abrir notebook interativo - Fase 2"
        Write-Host "========================================" -ForegroundColor Cyan
        foreach ($key in ($notebooksInterativos.Keys | Sort-Object)) {
            Write-Host ("  [{0}] {1}" -f $key, $notebooksInterativos[$key].Nome)
        }
        Write-Host "  [B] Voltar"
        Write-Host ""
        $choice = (Read-Host "Escolha uma opção").Trim()
        if ($choice -match '^[Bb]$') {
            return
        }
        if (-not $notebooksInterativos.ContainsKey($choice)) {
            Write-Host "Opção inválida." -ForegroundColor Red
            continue
        }
        try {
            Start-InteractiveNotebook -Notebook $notebooksInterativos[$choice]
        }
        catch {
            Write-Host $_.Exception.Message -ForegroundColor Red
        }
        return
    }
}

function Show-NotebookMenu {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Estudo
    )

    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ("  {0}" -f $Estudo.Nome)
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host ""

    for ($i = 0; $i -lt $Estudo.Notebooks.Count; $i++) {
        $numero = $i + 1
        Write-Host ("  [{0}] {1}" -f $numero, $Estudo.Notebooks[$i])
    }

    Write-Host ""
    Write-Host "  [T] Executar todos (na ordem)"
    Write-Host "  [B] Voltar"
    Write-Host "  [S] Sair"
    Write-Host ""
}

function Invoke-Notebook {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Estudo,
        [Parameter(Mandatory = $true)]
        [string]$Nome
    )

    $caminho = Join-Path $raiz (Join-Path $Estudo.Pasta $Nome)
    if (-not (Test-Path $caminho)) {
        throw "Notebook não encontrado: $caminho"
    }

    Write-Host ""
    Write-Host "Executando: $($Estudo.Nome) / $Nome ..." -ForegroundColor Yellow
    Write-Host ""

    & $venvPython $executor --skip-interactive $caminho
    if ($LASTEXITCODE -ne 0) {
        throw "Falha ao executar o notebook: $Nome"
    }

    Write-Host ""
    Write-Host "Concluído: $Nome" -ForegroundColor Green
}

function Invoke-AllNotebooks {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Estudo
    )

    $total = $Estudo.Notebooks.Count
    for ($i = 0; $i -lt $total; $i++) {
        $atual = $i + 1
        Write-Host ""
        Write-Host "===== Notebook $atual de $total =====" -ForegroundColor Cyan
        Invoke-Notebook -Estudo $Estudo -Nome $Estudo.Notebooks[$i]
    }

    Write-Host ""
    Write-Host "Todos os notebooks foram executados com sucesso." -ForegroundColor Green
}

function Invoke-AllEstudos {
    $ordem = $estudos.Keys | Sort-Object
    $totalEstudos = $ordem.Count
    $estudoAtual = 0

    foreach ($chave in $ordem) {
        $estudoAtual++
        $estudo = $estudos[$chave]
        Write-Host ""
        Write-Host "===== Estudo $estudoAtual de $totalEstudos : $($estudo.Nome) =====" -ForegroundColor Magenta
        Invoke-AllNotebooks -Estudo $estudo
    }

    Write-Host ""
    Write-Host "Todos os estudos foram executados com sucesso." -ForegroundColor Green
}

function Invoke-EstudoLoop {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Estudo
    )

    while ($true) {
        Show-NotebookMenu -Estudo $Estudo
        $escolha = (Read-Host "Escolha uma opção").Trim()

        if ($escolha -match '^[Ss]$') {
            Write-Host "Encerrado."
            break
        }

        if ($escolha -match '^[Bb]$') {
            break
        }

        $executou = $false

        try {
            if ($escolha -match '^[Tt]$') {
                Invoke-AllNotebooks -Estudo $Estudo
                $executou = $true
            }
            elseif ($escolha -match '^\d+$') {
                $indice = [int]$escolha - 1
                if ($indice -lt 0 -or $indice -ge $Estudo.Notebooks.Count) {
                    Write-Host "Opção inválida. Escolha de 1 a $($Estudo.Notebooks.Count)." -ForegroundColor Red
                    continue
                }
                Invoke-Notebook -Estudo $Estudo -Nome $Estudo.Notebooks[$indice]
                $executou = $true
            }
            else {
                Write-Host "Opção inválida." -ForegroundColor Red
                continue
            }
        }
        catch {
            Write-Host ""
            Write-Host $_.Exception.Message -ForegroundColor Red
            Write-Host "A execução foi interrompida." -ForegroundColor Red
        }

        if (-not $executou) {
            continue
        }

        Write-Host ""
        $outra = (Read-Host "Deseja executar outro notebook? (S/N)").Trim()
        if ($outra -notmatch '^[Ss]$') {
            break
        }
    }
}

while ($true) {
    Show-EstudoMenu
    $estudoEscolha = (Read-Host "Escolha uma opção").Trim()

    if ($estudoEscolha -match '^[Ss]$') {
        Write-Host "Encerrado."
        break
    }

    if ($estudoEscolha -match '^[Tt]$') {
        try {
            Invoke-AllEstudos
        }
        catch {
            Write-Host ""
            Write-Host $_.Exception.Message -ForegroundColor Red
            Write-Host "A execução foi interrompida." -ForegroundColor Red
        }

        Write-Host ""
        $outroEstudo = (Read-Host "Deseja escolher outro estudo? (S/N)").Trim()
        if ($outroEstudo -notmatch '^[Ss]$') {
            Write-Host "Encerrado."
            break
        }
        continue
    }

    if ($estudoEscolha -eq "3") {
        Invoke-ModelOptimizationMenu
        continue
    }
    if ($estudoEscolha -eq "4") {
        Invoke-InteractiveNotebookMenu
        continue
    }

    if (-not $estudos.ContainsKey($estudoEscolha)) {
        Write-Host "Opção inválida." -ForegroundColor Red
        continue
    }

    Invoke-EstudoLoop -Estudo $estudos[$estudoEscolha]

    Write-Host ""
    $outroEstudo = (Read-Host "Deseja escolher outro estudo? (S/N)").Trim()
    if ($outroEstudo -notmatch '^[Ss]$') {
        Write-Host "Encerrado."
        break
    }
}
