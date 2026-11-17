<#
Enchaine l'etape gmrio_to_snac_s pour chaque scenario SSP present dans
data\4-PlaneFR\2-shocks\3-SSPs_2050, en adaptant les chemins (scenario)
dans gmrio_to_snac_S_SSPs.json avant chaque run.

Suppose que l'etape engine a deja ete executee pour ces scenarios
(cf. run_engine_batch_SSPs.ps1) : path_in.accounts pointe vers
.\4-PlaneFR\Outputs\World\<scenario>.

Les sorties sont ecrites dans
.\4-PlaneFR\Outputs_inter\gmrio_to_snac_s\SSPs\<scenario>, puis leurs
F_x_dom sont remplaces par ceux de la sortie de reference
(SSPs\$referenceScenarioName).

A executer depuis l'environnement matmat-ademe (pipeline porte dans
src/matmat/workflows/pipelines/accounts/gmrio_to_snac_s).

-Suffix (ex. "_agri") : lit les chocs dans 3-SSPs_2050<Suffix>, les accounts
dans Outputs\World\<scenario><Suffix> et ecrit dans SSPs<Suffix>\<scenario><Suffix>.
La sortie de reference (F_x_dom) est lue dans SSPs<Suffix>\<ReferenceScenarioName> :
si ce scenario fait partie des chocs (ex. Base_year_agri), il est traite en
premier et ses F_x_dom ne sont pas remplaces.
#>

param(
    # Suffixe des dossiers de chocs et de sortie ("" par defaut, ex. "_agri")
    [string]$Suffix = "",
    # Sortie gmrio_to_snac_s de reference dont les F_x_dom sont recopies dans
    # chaque scenario (les intensites domestiques France ne sont pas choquees)
    [string]$ReferenceScenarioName = "Base_year_2015"
)

$projectDir = Split-Path -Parent $PSScriptRoot

$gmrioSettingsPath = Join-Path $projectDir "data\4-PlaneFR\Settings\2-gmrio_to_snac_s\gmrio_to_snac_S_SSPs.json"
$shocksDir         = Join-Path $projectDir "data\4-PlaneFR\2-shocks\3-SSPs_2050$Suffix"

function Add-Suffix {
    param([string]$Name)
    # Ajoute $Suffix au nom s'il ne s'y termine pas deja
    if ($Name.EndsWith($Suffix)) { return $Name }
    return "$Name$Suffix"
}

Set-Location $projectDir

# Decomposition des chemins initiaux :
# - accounts : "...\Outputs\World\<scenario>" -> dossier parent
# - path_out : "...\SSPs<suffixe>\<scenario>" -> dossier parent, ramene a
#   "...\SSPs" (le JSON garde le dernier chemin utilise, suffixe ou non)
$initialSettings = Get-Content -Raw -Path $gmrioSettingsPath | ConvertFrom-Json
$accountsBaseDir = Split-Path -Parent $initialSettings.path_in.accounts
$gmrioOutBaseDir = [regex]::Replace((Split-Path -Parent $initialSettings.path_out), "_agri$", "") + $Suffix

# Les chemins des settings sont relatifs au dossier data
$dataDir = Join-Path $projectDir "data"

function Set-GmrioToSnacSSettings {
    param(
        # Nom du sous-dossier de scenario (ex. "SSP2_2050")
        [string]$ScenarioName
    )

    $settings = Get-Content -Raw -Path $gmrioSettingsPath | ConvertFrom-Json

    $settings.path_in.accounts = Join-Path $accountsBaseDir (Add-Suffix $ScenarioName)
    $settings.path_out         = Join-Path $gmrioOutBaseDir (Add-Suffix $ScenarioName)

    $json = $settings | ConvertTo-Json -Depth 10
    # Ecriture en UTF-8 sans BOM pour eviter les erreurs de parsing JSON cote Python
    [System.IO.File]::WriteAllText($gmrioSettingsPath, $json, (New-Object System.Text.UTF8Encoding($false)))
}

# === Etape gmrio_to_snac_s : un run par scenario present dans
# data\4-PlaneFR\2-shocks\3-SSPs_2050 ===

# Scenario de reference en premier : ses F_x_dom servent aux suivants
$scenarios = Get-ChildItem -Path $shocksDir -Directory | Select-Object -ExpandProperty Name |
    Sort-Object { (Add-Suffix $_) -ne $ReferenceScenarioName }, { $_ }
$failed    = $false

foreach ($scenario in $scenarios) {
    Write-Host "=== $scenario : gmrio_to_snac_s ===" -ForegroundColor Cyan

    Set-GmrioToSnacSSettings -ScenarioName $scenario

    uv run python -m matmat.cli -p gmrio_to_snac_s -st $gmrioSettingsPath --no_confirm

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Echec gmrio_to_snac_s pour $scenario (code $LASTEXITCODE)" -ForegroundColor Red
        $failed = $true
        continue
    }

    # === Remplacement des F_x_dom par ceux de la sortie de reference ===
    if ((Add-Suffix $scenario) -eq $ReferenceScenarioName) {
        continue
    }
    $refDir   = Join-Path $dataDir (Join-Path $gmrioOutBaseDir $ReferenceScenarioName)
    $scenDir  = Join-Path $dataDir (Join-Path $gmrioOutBaseDir (Add-Suffix $scenario))
    $settings = Get-Content -Raw -Path $gmrioSettingsPath | ConvertFrom-Json

    Write-Host "--- $scenario : remplacement des F_x_dom par ceux de $refDir ---" -ForegroundColor Cyan

    foreach ($ext in @($settings.extension_names)) {
        $src = Join-Path $refDir  "extensions\dom_$ext\F_x_dom.pkl"
        $dst = Join-Path $scenDir "extensions\dom_$ext\F_x_dom.pkl"
        if (-not (Test-Path $src)) {
            Write-Host "F_x_dom de reference introuvable : $src" -ForegroundColor Red
            $failed = $true
            continue
        }
        Copy-Item -Path $src -Destination $dst -Force
    }
}

if ($failed) {
    exit 1
}
