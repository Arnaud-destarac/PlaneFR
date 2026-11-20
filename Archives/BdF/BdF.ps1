# Chaîne complète BdF, pour chaque scénario du groupe $Group de
# Module_PlaneFR/data/Outputs_MatMat/scenario_groups.json :
#
# 1. export_impacts_coicop.py       : écrit impacts.csv (scénario passé via PLANEFR_SCENARIO)
# 2. imputation_impacts.ipynb       : exécuté en place, produit impacts_facteurs.csv et sa légende
# 3. agregation_impacts_facteurs.py : produit impacts_facteurs_agrégé.xlsx,
#                                     renommé ensuite en impacts_facteurs_agrégé_{groupe}_{scénario}.xlsx
#
# Les scénarios et leurs noms sont ceux du groupe dans scenario_groups.json
# (ex. groupe "BdF" : "2019" et "S1+" -> impacts_facteurs_agrégé_BdF_2019.xlsx et
# impacts_facteurs_agrégé_BdF_S1+.xlsx). C'est exactement ce que cherche
# planefr_lib.io (bdf_file pour le scénario 2050, bdf_comparison_file pour
# l'année de base) avec scenario_group = $Group, sans renommage manuel.
# Le scénario lu est la copie du groupe (data/Outputs_MatMat/<version>/<groupe>/<nom>),
# construite par build_scenario_groups.py.

param(
    [string]$Group = "BdF",
    [string]$Version = $(if ($env:PLANEFR_EXIOBASE_VERSION) { $env:PLANEFR_EXIOBASE_VERSION } else { "3.11.2" })
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$OutputsDir = Join-Path $PSScriptRoot "..\..\Module_PlaneFR\data\Outputs_MatMat"
$GroupsFile = Join-Path $OutputsDir "scenario_groups.json"
$Groups = (Get-Content -Raw -Encoding UTF8 $GroupsFile | ConvertFrom-Json).groups
if (-not ($Groups.PSObject.Properties.Name -contains $Group)) {
    throw "Groupe '$Group' absent de $GroupsFile (groupes : $($Groups.PSObject.Properties.Name -join ', '))"
}
$Names = $Groups.$Group.PSObject.Properties.Name

function Invoke-Step($Name, [scriptblock]$Command) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'étape '$Name' (code $LASTEXITCODE)" }
}

$Outputs = @()
foreach ($Name in $Names) {
    $Scenario = "$Group/$Name"
    if (-not (Test-Path (Join-Path $OutputsDir "$Version\$Scenario\extensions"))) {
        throw "Dossier $Scenario introuvable dans $OutputsDir\$Version (lancer build_scenario_groups.py)"
    }
    Write-Host "`n##### Scénario : $Scenario (source $($Groups.$Group.$Name)) #####" -ForegroundColor Magenta
    $env:PLANEFR_SCENARIO = $Scenario

    Invoke-Step "1/3 export_impacts_coicop ($Scenario)" {
        py -3 export_impacts_coicop.py
    }

    Invoke-Step "2/3 imputation_impacts ($Scenario)" {
        py -3 -m jupyter nbconvert --to notebook --execute --inplace imputation_impacts.ipynb
    }

    Invoke-Step "3/3 agregation_impacts_facteurs ($Scenario)" {
        py -3 agregation_impacts_facteurs.py
    }

    $Target = "impacts_facteurs_agrégé_${Group}_$Name.xlsx"
    Move-Item -Force "impacts_facteurs_agrégé.xlsx" $Target
    Write-Host "Sauvegardé : $Target" -ForegroundColor Green
    $Outputs += $Target
}

Remove-Item Env:PLANEFR_SCENARIO
Write-Host "`nTerminé. Fichiers produits :" -ForegroundColor Green
$Outputs | ForEach-Object { Write-Host "  $_" }
