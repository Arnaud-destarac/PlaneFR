<#
Enchaine l'etape update_calib_fr pour chaque annee de base (2015, 2019), en
adaptant les chemins (annee) dans le fichier de settings JSON avant chaque
run.

Le pipeline "update_calib_fr" n'est pas (encore) porte dans matmat-ademe
(src/matmat/workflows/pipelines/accounts/ ne contient que "aggregation" et
"gmrio_to_snac_s" ; -p update_calib_fr n'existe pas dans son cli.py). Il
n'existe que dans l'ancien wheel installe pour le projet datapack
(matmat==0.9.0b0). Ce script edite donc le fichier de settings situe dans
matmat-ademe, mais lance la commande depuis le dossier datapack (frere de
matmat-ademe), pour utiliser son environnement/CLI.

La version Exiobase (-Version, 3.11.2 par defaut) est reportee dans le chemin
newer_accounts (sortie gmrio_to_snac_s) ; older_accounts (comptes France de
reference) n'est pas modifie.
#>

param(
    # Version Exiobase a utiliser (ex. "3.11.2")
    [string]$Version = "3.11.2"
)

# Chemins calcules relativement a l'emplacement de ce script, pour que le
# script fonctionne quel que soit le repertoire courant depuis lequel il est
# invoque.
$matmatAdemeDir = Split-Path -Parent $PSScriptRoot
$matmatRootDir  = Split-Path -Parent $matmatAdemeDir
$datapackDir    = Join-Path $matmatRootDir "datapack"

$calibSettingsPath = Join-Path $matmatAdemeDir "data\4-PlaneFR\Settings\3-update_calib_fr\update_calib_fr_PlaneFR.json"

$baseYears = 2015, 2019

function Set-YearInPath {
    param(
        [string]$Path,
        [int]$Year
    )
    # Remplace la derniere occurrence "_20xx" (annee) par "_$Year", en
    # conservant le reste du chemin (version, sous-dossiers) inchange.
    return [regex]::Replace($Path, "_20\d{2}(?!.*_20\d{2})", "_$Year")
}

function Set-VersionInPath {
    param(
        [string]$Path,
        [string]$Version
    )
    # Remplace toute occurrence d'un numero de version Exiobase (ex. 3.10.2)
    # par la version demandee, en conservant le reste du chemin inchange.
    return [regex]::Replace($Path, "\d+\.\d+\.\d+", $Version)
}

function Set-UpdateCalibFrSettings {
    param(
        [int]$Year
    )

    $settings = Get-Content -Raw -Path $calibSettingsPath | ConvertFrom-Json

    $settings.path_in.newer_accounts = Set-YearInPath -Path $settings.path_in.newer_accounts -Year $Year
    $settings.path_in.newer_accounts = Set-VersionInPath -Path $settings.path_in.newer_accounts -Version $Version
    $settings.path_in.older_accounts = Set-YearInPath -Path $settings.path_in.older_accounts -Year $Year
    $settings.path_out                = Set-YearInPath -Path $settings.path_out -Year $Year

    $json = $settings | ConvertTo-Json -Depth 10
    # Ecriture en UTF-8 sans BOM pour eviter les erreurs de parsing JSON cote Python
    [System.IO.File]::WriteAllText($calibSettingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

Set-Location $datapackDir

foreach ($year in $baseYears) {
    Write-Host "=== base_year=$year : update_calib_fr ===" -ForegroundColor Cyan

    Set-UpdateCalibFrSettings -Year $year

    uv run python -m matmat.cli -p update_calib_fr -st $calibSettingsPath --no_confirm

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Echec update_calib_fr pour base_year=$year (code $LASTEXITCODE)" -ForegroundColor Red
    }
}
