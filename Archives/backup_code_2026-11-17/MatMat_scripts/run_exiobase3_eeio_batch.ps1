<#
Lance plusieurs fois la commande exiobase3_eeio en faisant varier
base_year et version dans le fichier de settings JSON.

La version Exiobase est passee en parametre (-Version, 3.11.2 par defaut).
#>

param(
    # Version Exiobase a utiliser (ex. "3.11.2")
    [string]$Version = "3.11.2"
)

$projectDir   = Split-Path -Parent $PSScriptRoot
$settingsPath = Join-Path $projectDir "data\4-PlaneFR\Settings\1-exiobase3_eeio\exiobase3_eeio_settings.json"

$baseYears = 2015, 2019
$versions  = $Version

Set-Location $projectDir

function Set-JsonValueAndSave {
    param(
        [int]$BaseYear,
        [string]$Version
    )

    $settings = Get-Content -Raw -Path $settingsPath | ConvertFrom-Json
    $settings.base_year = $BaseYear
    $settings.version   = $Version
    $settings.path_out  = ".\4-PlaneFR\1-accounts\Exiobase\${Version}_${BaseYear}"

    $json = $settings | ConvertTo-Json -Depth 10
    # Ecriture en UTF-8 sans BOM pour eviter les erreurs de parsing JSON cote Python
    [System.IO.File]::WriteAllText($settingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

foreach ($baseYear in $baseYears) {
    foreach ($version in $versions) {
        Write-Host "=== base_year=$baseYear / version=$version ===" -ForegroundColor Cyan

        Set-JsonValueAndSave -BaseYear $baseYear -Version $version

        uv run python -m matmat.cli -a exiobase3_eeio -st $settingsPath --no_confirm

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Echec pour base_year=$baseYear / version=$version (code $LASTEXITCODE)" -ForegroundColor Red
        }
    }
}

# à insérer si besoin

# Lance le moteur eeio (pour la version $Version), en adaptant
# path_in et path_out a chaque base_year dans settings_engine_World.json.
$engineSettingsPath = Join-Path $projectDir "data\4-PlaneFR\Settings\4-engine\settings_engine_World.json"
$engineVersion = $Version

function Set-EngineJsonValueAndSave {
    param(
        [int]$BaseYear
    )

    $settings = Get-Content -Raw -Path $engineSettingsPath | ConvertFrom-Json
    $settings.path_in.accounts = ".\4-PlaneFR\1-accounts\Exiobase\${engineVersion}_${BaseYear}\${engineVersion}_pxp_${BaseYear}"
    $settings.path_out          = ".\4-PlaneFR\Outputs\World\Base_year_${BaseYear}"

    $json = $settings | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($engineSettingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

foreach ($baseYear in $baseYears) {
    Write-Host "=== eeio: base_year=$baseYear / version=$engineVersion ===" -ForegroundColor Cyan

    Set-EngineJsonValueAndSave -BaseYear $baseYear

    uv run -m matmat.cli -e eeio -st $engineSettingsPath --no_confirm

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Echec eeio pour base_year=$baseYear (code $LASTEXITCODE)" -ForegroundColor Red
    }
}
