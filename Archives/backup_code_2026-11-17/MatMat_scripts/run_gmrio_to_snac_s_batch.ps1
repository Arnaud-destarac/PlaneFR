<#
Enchaine l'etape gmrio_to_snac_s pour chaque annee de base (2015, 2019),
pour la version d'Exiobase passee en parametre (-Version, 3.11.2 par
defaut), en adaptant les chemins (annee, version) dans le fichier de
settings JSON avant chaque run.

Recree ensuite les sous-dossiers "extensions" de chaque scenario de
data\4-PlaneFR\2-shocks\1-transitions_2050 (dom_<ext> pour chaque
extension_name de gmrio_to_snac_s, avec detail_levels.xlsx et info.json).

A executer depuis l'environnement matmat-ademe (pipeline porte dans
src/matmat/workflows/pipelines/accounts/gmrio_to_snac_s).
#>

param(
    # Version Exiobase a utiliser (ex. "3.11.2")
    [string]$Version = "3.11.2"
)

$projectDir = Split-Path -Parent $PSScriptRoot

$gmrioSettingsPath = Join-Path $projectDir "data\4-PlaneFR\Settings\2-gmrio_to_snac_s\gmrio_to_snac_S_FR.json"

$baseYears = 2015, 2019
$versions  = $Version

Set-Location $projectDir

function Set-YearInPath {
    param(
        [string]$Path,
        [int]$Year
    )
    # Remplace toutes les occurrences "_20xx" (annee) par "_$Year", pour que
    # le dossier version_annee et le sous-dossier version_pxp_annee restent
    # coherents entre eux.
    return [regex]::Replace($Path, "_20\d{2}", "_$Year")
}

function Set-VersionInPath {
    param(
        [string]$Path,
        [string]$Version
    )
    # Remplace toute occurrence d'un numero de version Exiobase (ex. 3.10.2,
    # 3.9.6) par la version demandee, en conservant le reste du chemin
    # (annee, sous-dossiers) inchange.
    return [regex]::Replace($Path, "\d+\.\d+\.\d+", $Version)
}

function Set-GmrioToSnacSSettings {
    param(
        [int]$Year,
        [string]$Version
    )

    $settings = Get-Content -Raw -Path $gmrioSettingsPath | ConvertFrom-Json

    $settings.path_in.accounts = Set-YearInPath -Path $settings.path_in.accounts -Year $Year
    $settings.path_in.accounts = Set-VersionInPath -Path $settings.path_in.accounts -Version $Version
    $settings.path_out          = Set-YearInPath -Path $settings.path_out -Year $Year
    $settings.path_out          = Set-VersionInPath -Path $settings.path_out -Version $Version

    $json = $settings | ConvertTo-Json -Depth 10
    # Ecriture en UTF-8 sans BOM pour eviter les erreurs de parsing JSON cote Python
    [System.IO.File]::WriteAllText($gmrioSettingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

foreach ($year in $baseYears) {
    foreach ($version in $versions) {
        Write-Host "=== base_year=$year / version=$version : gmrio_to_snac_s ===" -ForegroundColor Cyan

        Set-GmrioToSnacSSettings -Year $year -Version $version

        uv run python -m matmat.cli -p gmrio_to_snac_s -st $gmrioSettingsPath --no_confirm

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Echec gmrio_to_snac_s pour base_year=$year / version=$version (code $LASTEXITCODE)" -ForegroundColor Red
        }
    }
}

# === Reconstruction des dossiers "extensions" des scenarios de chocs ===
# Pour chaque scenario de $shocksDir, supprime le sous-dossier "extensions"
# puis le recree avec un dossier dom_<ext> par extension_name de
# gmrio_to_snac_s, contenant le detail_levels.xlsx et l'info.json
# correspondants (cf. rebuild_shock_extensions.py). La sortie gmrio_to_snac_s
# utilisee est celle de la base_year du scenario (system\info.json) et de la
# version $Version.
$shocksDir = Join-Path $projectDir "data\4-PlaneFR\2-shocks\1-transitions_2050"
$dataDir   = Join-Path $projectDir "data"

Write-Host "=== Reconstruction des extensions des chocs ($shocksDir) ===" -ForegroundColor Cyan

$settings       = Get-Content -Raw -Path $gmrioSettingsPath | ConvertFrom-Json
$extensionNames = @($settings.extension_names)
$failed         = $false

foreach ($scenarioDir in Get-ChildItem -Path $shocksDir -Directory) {
    $systemInfoPath = Join-Path $scenarioDir.FullName "system\info.json"
    if (-not (Test-Path $systemInfoPath)) {
        Write-Host "Scenario $($scenarioDir.Name) ignore : $systemInfoPath introuvable" -ForegroundColor Yellow
        continue
    }
    $baseYear = [int](Get-Content -Raw -Path $systemInfoPath | ConvertFrom-Json).base_year

    $gmrioOutDir = Join-Path $dataDir (Set-YearInPath -Path $settings.path_out -Year $baseYear)
    $missing = $extensionNames | Where-Object {
        -not (Test-Path (Join-Path $gmrioOutDir "extensions\dom_$_\detail_levels.xlsx"))
    }
    if ($missing) {
        # On ne supprime rien si la sortie gmrio_to_snac_s est incomplete
        Write-Host "Scenario $($scenarioDir.Name) ignore : extensions absentes de $gmrioOutDir : $($missing -join ', ')" -ForegroundColor Red
        $failed = $true
        continue
    }

    $extensionsDir = Join-Path $scenarioDir.FullName "extensions"
    if (Test-Path $extensionsDir) {
        Remove-Item -Path $extensionsDir -Recurse -Force
    }

    uv run python (Join-Path $PSScriptRoot "rebuild_shock_extensions.py") `
        --scenario-dir $scenarioDir.FullName `
        --gmrio-dir $gmrioOutDir `
        --extensions $extensionNames

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Echec de la reconstruction des extensions pour $($scenarioDir.Name) (code $LASTEXITCODE)" -ForegroundColor Red
        $failed = $true
    }
}

if ($failed) {
    exit 1
}
