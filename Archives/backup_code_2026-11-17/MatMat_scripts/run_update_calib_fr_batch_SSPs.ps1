<#
Enchaine l'etape update_calib_fr pour chaque scenario present dans
data\4-PlaneFR\Outputs_inter\gmrio_to_snac_s\SSPs (sorties gmrio_to_snac_s,
cf. run_gmrio_to_snac_s_batch_SSPs.ps1), en adaptant les chemins dans le
fichier de settings JSON avant chaque run :
- newer_accounts : dossier du scenario s'il contient directement system\
  (ex. Base_year_2015), sinon son sous-dossier <Version>_pxp_<annee>
  (ex. SSP1_2050\3.11.2_pxp_2019) ;
- older_accounts : annee remplacee par la base_year du scenario (lue dans
  system\info.json), ex. 3.10.2_France_2015 ou 3.10.2_France_2019 ;
- path_out : .\4-PlaneFR\Outputs\France\<scenario>.

Le pipeline "update_calib_fr" n'est pas (encore) porte dans matmat-ademe
(src/matmat/workflows/pipelines/accounts/ ne contient que "aggregation" et
"gmrio_to_snac_s" ; -p update_calib_fr n'existe pas dans son cli.py). Il
n'existe que dans l'ancien wheel installe pour le projet datapack
(matmat==0.9.0b0). Ce script edite donc le fichier de settings situe dans
matmat-ademe, mais lance la commande depuis le dossier datapack (frere de
matmat-ademe), pour utiliser son environnement/CLI.

-Suffix (ex. "_agri") : lit Outputs_inter\gmrio_to_snac_s\SSPs<Suffix> et
ecrit dans Outputs\France<Suffix>.
#>

param(
    # Version Exiobase a utiliser (ex. "3.11.2") : selectionne le
    # sous-dossier <Version>_pxp_<annee> des scenarios qui en ont un
    [string]$Version = "3.11.2",
    # Suffixe des dossiers d'entree et de sortie ("" par defaut, ex. "_agri")
    [string]$Suffix = ""
)

# Chemins calcules relativement a l'emplacement de ce script, pour que le
# script fonctionne quel que soit le repertoire courant depuis lequel il est
# invoque.
$matmatAdemeDir = Split-Path -Parent $PSScriptRoot
$matmatRootDir  = Split-Path -Parent $matmatAdemeDir
$datapackDir    = Join-Path $matmatRootDir "datapack"

$calibSettingsPath = Join-Path $matmatAdemeDir "data\4-PlaneFR\Settings\3-update_calib_fr\update_calib_fr_PlaneFR_SSPs.json"

# Les chemins des settings sont relatifs au dossier data
$dataDir = Join-Path $matmatAdemeDir "data"

# Chemins (relatifs a data) des sorties gmrio_to_snac_s par scenario et des
# sorties update_calib_fr
$newerBaseDir = ".\4-PlaneFR\Outputs_inter\gmrio_to_snac_s\SSPs$Suffix"
$outBaseDir   = ".\4-PlaneFR\Outputs\France$Suffix"

function Set-YearInPath {
    param(
        [string]$Path,
        [int]$Year
    )
    # Remplace la derniere occurrence "_20xx" (annee) par "_$Year", en
    # conservant le reste du chemin (version, sous-dossiers) inchange.
    return [regex]::Replace($Path, "_20\d{2}(?!.*_20\d{2})", "_$Year")
}

function Get-NewerAccountsDir {
    param(
        # Dossier du scenario (chemin absolu)
        [string]$ScenarioDir
    )
    # Renvoie le chemin relatif (au dossier du scenario) des comptes
    # gmrio_to_snac_s : "" si system\ est directement dans le scenario, sinon
    # le sous-dossier <Version>_pxp_<annee> qui contient system\. $null si
    # aucun n'est trouve.
    if (Test-Path (Join-Path $ScenarioDir "system\info.json")) {
        return ""
    }
    $sub = Get-ChildItem -Path $ScenarioDir -Directory -Filter "$($Version)_pxp_*" |
        Where-Object { Test-Path (Join-Path $_.FullName "system\info.json") } |
        Select-Object -First 1
    if ($null -eq $sub) {
        return $null
    }
    return $sub.Name
}

function Set-UpdateCalibFrSettings {
    param(
        [string]$NewerAccounts,
        [string]$PathOut,
        [int]$Year
    )

    $settings = Get-Content -Raw -Path $calibSettingsPath | ConvertFrom-Json

    $settings.path_in.newer_accounts = $NewerAccounts
    $settings.path_in.older_accounts = Set-YearInPath -Path $settings.path_in.older_accounts -Year $Year
    $settings.path_out               = $PathOut

    $json = $settings | ConvertTo-Json -Depth 10
    # Ecriture en UTF-8 sans BOM pour eviter les erreurs de parsing JSON cote Python
    [System.IO.File]::WriteAllText($calibSettingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

# === Etape update_calib_fr : un run par scenario present dans
# data\4-PlaneFR\Outputs_inter\gmrio_to_snac_s\SSPs ===

$scenarioDirs = Get-ChildItem -Path (Join-Path $dataDir $newerBaseDir) -Directory

Set-Location $datapackDir

foreach ($scenarioDir in $scenarioDirs) {
    $scenario = $scenarioDir.Name

    $sub = Get-NewerAccountsDir -ScenarioDir $scenarioDir.FullName
    if ($null -eq $sub) {
        Write-Host "Scenario $scenario ignore : aucun system\info.json (ni dans un sous-dossier $($Version)_pxp_*)" -ForegroundColor Yellow
        continue
    }

    $newerAccounts = Join-Path $newerBaseDir $scenario
    if ($sub) {
        $newerAccounts = Join-Path $newerAccounts $sub
    }

    $systemInfoPath = Join-Path $dataDir (Join-Path $newerAccounts "system\info.json")
    $baseYear       = [int](Get-Content -Raw -Path $systemInfoPath | ConvertFrom-Json).base_year

    Write-Host "=== $scenario (base_year=$baseYear) : update_calib_fr ===" -ForegroundColor Cyan

    Set-UpdateCalibFrSettings -NewerAccounts $newerAccounts -PathOut (Join-Path $outBaseDir $scenario) -Year $baseYear

    uv run python -m matmat.cli -p update_calib_fr -st $calibSettingsPath --no_confirm

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Echec update_calib_fr pour $scenario (code $LASTEXITCODE)" -ForegroundColor Red
    }
}
