<#
Variante "_agri" de run_all_SSPs.ps1 : enchaine les memes etapes sur les
chocs agricoles de 2-shocks\1-transitions_2050_agri et
2-shocks\3-SSPs_2050_agri. Chaque dossier genere prend le suffixe "_agri" :
1. generate_shock_agri         (generate_shock_agri.py : regenere
                                1-transitions_2050_agri et 3-SSPs_2050_agri a
                                partir de 1-transitions_2050 et 3-SSPs_2050,
                                plus un scenario Base_year_agri dans chacun)
2. engine                      (run_engine_batch_SSPs.ps1 -Suffix _agri :
                                chocs de 3-SSPs_2050_agri, sorties dans
                                Outputs\World\<scenario>_agri)
3. gmrio_to_snac_s             (run_gmrio_to_snac_s_batch_SSPs.ps1 -Suffix _agri :
                                sorties dans Outputs_inter\gmrio_to_snac_s\SSPs_agri ;
                                F_x_dom de reference lus dans SSPs_agri\Base_year_agri)
4. update_calib_fr             (run_update_calib_fr_batch_SSPs.ps1 -Suffix _agri :
                                sorties dans Outputs\France_agri)
5. engine France               (run_engine_batch.ps1 -Suffix _agri : chocs de
                                1-transitions_2050_agri sur chaque jeu d'accounts
                                de Outputs\France_agri, sorties dans
                                Module_PlaneFR/data/<version>/<groupe>_agri/ ;
                                accounts Base_year_agri -> groupe no_SSP_agri,
                                dont la sortie Base_year_agri est copiee dans
                                chaque autre groupe comme reference)
6. reformat_d_cba_monde_europe (reformat_d_cba_monde_europe.py)
7. reformat_d_cba_k            (reformat_d_cba_k.py)
8. compute_ds_x_dom            (compute_ds_x_dom.py --only-f-y-tot : l'etape 6
                                remet F_Y_tot a sa valeur de l'annee de base dans
                                TOUS les dossiers de scenario, "_agri" ou non ;
                                les chocs F_Y_tot y sont donc reappliques. Les
                                dS_x_dom ne sont pas recalcules.)

generate_shock_agri.py lit les chocs non agricoles 1-transitions_2050 et
3-SSPs_2050 : run_all_SSPs.ps1 (qui genere 3-SSPs_2050) doit avoir ete
execute avant si ces chocs ont change.

S'arrete si une etape se termine avec un code d'erreur (cf. run_all_SSPs.ps1).
#>

param(
    # Version Exiobase a utiliser par update_calib_fr et les scripts de
    # reformatage (ex. "3.11.2")
    [string]$Version = "3.11.2"
)

$env:PLANEFR_EXIOBASE_VERSION = $Version

$suffix = "_agri"

# Scenario "annee de base" cree par generate_shock_agri.py (BASE_YEAR_SCENARIO)
$baseYearScenario = "Base_year$suffix"

$matmatAdemeDir = Split-Path -Parent $PSScriptRoot

$steps = @(
    @{ Name = "generate_shock_agri";         Script = "generate_shock_agri.py" },
    @{ Name = "engine";                      Script = "run_engine_batch_SSPs.ps1";          Args = @{ Suffix = $suffix } },
    @{ Name = "gmrio_to_snac_s";             Script = "run_gmrio_to_snac_s_batch_SSPs.ps1"; Args = @{ Suffix = $suffix; ReferenceScenarioName = $baseYearScenario } },
    @{ Name = "update_calib_fr";             Script = "run_update_calib_fr_batch_SSPs.ps1"; Args = @{ Version = $Version; Suffix = $suffix } },
    @{ Name = "engine France";               Script = "run_engine_batch.ps1";               Args = @{ Version = $Version; Suffix = $suffix } },
    @{ Name = "reformat_d_cba_monde_europe"; Script = "reformat_d_cba_monde_europe.py" },
    @{ Name = "reformat_d_cba_k";            Script = "reformat_d_cba_k.py" },
    @{ Name = "compute_ds_x_dom (F_Y_tot)";  Script = "compute_ds_x_dom.py";               PyArgs = @("--only-f-y-tot") }
)

foreach ($step in $steps) {
    Write-Host "########## Etape $($step.Name) ($suffix) ##########" -ForegroundColor Magenta

    $scriptPath = Join-Path $PSScriptRoot $step.Script
    $global:LASTEXITCODE = 0
    if ($step.Script.EndsWith(".py")) {
        Set-Location $matmatAdemeDir
        # Arguments du script Python (aucun par defaut)
        $pyArgs = if ($step.PyArgs) { $step.PyArgs } else { @() }
        uv run python $scriptPath @pyArgs
    }
    else {
        $scriptArgs = $step.Args
        & $scriptPath @scriptArgs
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Arret : echec de l'etape $($step.Name) (code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "########## Toutes les etapes SSPs ($suffix) sont terminees ##########" -ForegroundColor Green
