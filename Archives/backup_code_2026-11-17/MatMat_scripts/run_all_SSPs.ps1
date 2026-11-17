<#
Enchaine les etapes SSPs dans l'ordre :
1. extract_S_SSPs              (extract_S_SSPs.py : ratios F, x, S 2050/2020
                                des scenarios IMAGE -> feuilles de
                                Matching_extensions_EXIOBASE_IMAGE.xlsx)
2. generate_shock_dS           (generate_shock_dS.py : dS_x_dom et dA dans
                                2-shocks\3-SSPs_2050\<SSP>_2050)
3. engine                      (run_engine_batch_SSPs.ps1)
4. gmrio_to_snac_s             (run_gmrio_to_snac_s_batch_SSPs.ps1)
5. update_calib_fr             (run_update_calib_fr_batch_SSPs.ps1)
6. engine France               (run_engine_batch.ps1 : baselines + chocs de
                                1-transitions_2050 sur chaque jeu d'accounts
                                de Outputs\France, sorties dans
                                Module_PlaneFR/data/<version>/<SSP|no_SSP>/)
7. reformat_d_cba_monde_europe (reformat_d_cba_monde_europe.py)
8. reformat_d_cba_k            (reformat_d_cba_k.py)
9. export_dS                   (export_dS.py : S_extraction.xlsx)
10. compute_ds_x_dom           (compute_ds_x_dom.py : dS_x_dom des chocs
                                manuels + application des chocs F_Y_tot)

Pas d'etape extract_f_y_tot : elle ne depend que d'Exiobase brut (F_Y_tot
World de Base_year_2015/2019), pas des scenarios ; elle doit avoir ete
lancee au moins une fois (run_pipeline_from_engine.ps1 ou a la main) pour
que reformat_d_cba_monde_europe trouve Outputs\World\Base_year_*\...\F_Y_tot.pkl.

Les etapes 7 et 8 (comme dans run_pipeline_from_engine.ps1) traitent tous les
scenarios de Module_PlaneFR/data/<version>, y compris ceux imbriques par SSP
(data/<version>/<SSP>/<scenario>). La version est transmise aux scripts
Python via la variable d'environnement PLANEFR_EXIOBASE_VERSION.

S'arrete si une etape se termine avec un code d'erreur (seul
run_gmrio_to_snac_s_batch_SSPs.ps1 renvoie exit 1 en cas d'echec parmi les
scripts .ps1 ; les autres signalent leurs echecs par scenario sans
interrompre la boucle).
#>

param(
    # Version Exiobase a utiliser par update_calib_fr et les scripts de
    # reformatage (ex. "3.11.2")
    [string]$Version = "3.11.2"
)

$env:PLANEFR_EXIOBASE_VERSION = $Version

$matmatAdemeDir = Split-Path -Parent $PSScriptRoot

$steps = @(
    @{ Name = "extract_S_SSPs";              Script = "extract_S_SSPs.py" },
    @{ Name = "generate_shock_dS";           Script = "generate_shock_dS.py" },
    @{ Name = "engine";                    Script = "run_engine_batch_SSPs.ps1";          Args = @{} },
    @{ Name = "gmrio_to_snac_s";             Script = "run_gmrio_to_snac_s_batch_SSPs.ps1"; Args = @{} },
    @{ Name = "update_calib_fr";             Script = "run_update_calib_fr_batch_SSPs.ps1"; Args = @{ Version = $Version } },
    @{ Name = "engine France";               Script = "run_engine_batch.ps1";               Args = @{ Version = $Version } },
    @{ Name = "reformat_d_cba_monde_europe"; Script = "reformat_d_cba_monde_europe.py" },
    @{ Name = "reformat_d_cba_k";            Script = "reformat_d_cba_k.py" },
    @{ Name = "export_dS";                   Script = "export_dS.py" },
    @{ Name = "compute_ds_x_dom";            Script = "compute_ds_x_dom.py" }
)

foreach ($step in $steps) {
    Write-Host "########## Etape $($step.Name) ##########" -ForegroundColor Magenta

    $scriptPath = Join-Path $PSScriptRoot $step.Script
    $global:LASTEXITCODE = 0
    if ($step.Script.EndsWith(".py")) {
        Set-Location $matmatAdemeDir
        uv run python $scriptPath
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

Write-Host "########## Toutes les etapes SSPs sont terminees ##########" -ForegroundColor Green
