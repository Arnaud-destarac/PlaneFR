# Chaîne complète BdF, pour chaque scénario de $Scenarios :
#
# 1. export_impacts_coicop.py       : écrit impacts.csv (scénario passé via PLANEFR_SCENARIO)
# 2. imputation_impacts.ipynb       : exécuté en place, produit impacts_facteurs.csv et sa légende
# 3. agregation_impacts_facteurs.py : produit impacts_facteurs_agrégé.xlsx,
#                                     renommé ensuite en impacts_facteurs_agrégé_{scenario}.xlsx

# Scénarios à traiter, chemins relatifs à DATA_DIR/3.11.2
$Scenarios = @(
    "SSP2_19_Sus_2050/S1_2050",
    "Base_year/2019_FR"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Step($Name, [scriptblock]$Command) {
    Write-Host "`n=== $Name ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Échec de l'étape '$Name' (code $LASTEXITCODE)" }
}

$Outputs = @()
foreach ($Scenario in $Scenarios) {
    # "SSP2_19_Sus_2050/S1_2050" -> "SSP2_19_Sus_2050_S1_2050"
    $Tag = ($Scenario -replace '[\\/]+', '_').Trim('_')
    Write-Host "`n##### Scénario : $Scenario #####" -ForegroundColor Magenta
    $env:PLANEFR_SCENARIO = $Scenario

    Invoke-Step "1/3 export_impacts_coicop ($Tag)" {
        py -3 export_impacts_coicop.py
    }

    Invoke-Step "2/3 imputation_impacts ($Tag)" {
        py -3 -m jupyter nbconvert --to notebook --execute --inplace imputation_impacts.ipynb
    }

    Invoke-Step "3/3 agregation_impacts_facteurs ($Tag)" {
        py -3 agregation_impacts_facteurs.py
    }

    $Target = "impacts_facteurs_agrégé_$Tag.xlsx"
    Move-Item -Force "impacts_facteurs_agrégé.xlsx" $Target
    Write-Host "Sauvegardé : $Target" -ForegroundColor Green
    $Outputs += $Target
}

Remove-Item Env:PLANEFR_SCENARIO
Write-Host "`nTerminé. Fichiers produits :" -ForegroundColor Green
$Outputs | ForEach-Object { Write-Host "  $_" }
