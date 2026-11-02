param(
    [string]$Destination = "src\db\models\dataset_cnes\hospital_supply_chain\inventory_data.csv",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$destinationPath = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $Destination))
$destinationDirectory = Split-Path -Parent $destinationPath

if ((Test-Path -LiteralPath $destinationPath) -and -not $Force) {
    throw "O arquivo já existe: $destinationPath. Use -Force para substituí-lo."
}

$archivePath = Join-Path ([System.IO.Path]::GetTempPath()) (
    "hospital-supply-chain-" + [Guid]::NewGuid().ToString("N") + ".zip"
)

try {
    Invoke-WebRequest `
        -Uri "https://www.kaggle.com/api/v1/datasets/download/vanpatangan/hospital-supply-chain" `
        -OutFile $archivePath

    Add-Type -AssemblyName System.IO.Compression.FileSystem
    $archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        $entry = $archive.GetEntry("inventory_data.csv")
        if ($null -eq $entry) {
            throw "O arquivo inventory_data.csv não foi encontrado no dataset Kaggle."
        }
        if (-not (Test-Path -LiteralPath $destinationDirectory)) {
            New-Item -ItemType Directory -Path $destinationDirectory -Force | Out-Null
        }
        $sourceStream = $entry.Open()
        try {
            $destinationStream = [System.IO.File]::Create($destinationPath)
            try {
                $sourceStream.CopyTo($destinationStream)
            }
            finally {
                $destinationStream.Dispose()
            }
        }
        finally {
            $sourceStream.Dispose()
        }
    }
    finally {
        $archive.Dispose()
    }

    $hash = (Get-FileHash -LiteralPath $destinationPath -Algorithm SHA256).Hash
    Write-Output "Inventário Kaggle salvo em: $destinationPath"
    Write-Output "SHA-256: $hash"
    Write-Output "Fonte: https://www.kaggle.com/datasets/vanpatangan/hospital-supply-chain (licença MIT)"
}
finally {
    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath
    }
}
