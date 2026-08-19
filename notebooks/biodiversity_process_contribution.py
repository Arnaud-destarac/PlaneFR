"""
Contribution des différents processus du système Terre (LP, facteurs de
caractérisation.xlsx) à l'empreinte du sous-processus "Biodiversity loss",
en % de l'empreinte totale -- une barre empilée à 100 % par scénario France
(scénario de base + scénarios de transition 2050, cf. config.get_scenario_folders).

ghg_combustion et ghg_emissions sont regroupés sous "Climate Change" ; les
autres LP sont renommés pour l'affichage : land_use -> "Land occupation",
water -> "Water Availability", biogeochemical -> "Eutrophication",
air_emissions -> "Other" (voir planefr_lib.plot_process_breakdown).

Génère figures/biodiversity_loss_process_contribution.png.
"""

import sys
from pathlib import Path

notebooks_dir = Path(__file__).resolve().parent
while not (notebooks_dir / "planefr_lib").exists() and notebooks_dir != notebooks_dir.parent:
    notebooks_dir = notebooks_dir.parent
if str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

import matplotlib.pyplot as plt

from planefr_lib import config, io, processing
from planefr_lib.plot_process_breakdown import create_subprocess_contribution_chart

SUBPROCESS_NAME = "Biodiversity loss"


def main():
    facteurs_carac_df = io.load_facteurs_carac()
    seuils_df = io.load_seuils()

    scenario_folders = config.get_scenario_folders(exclude_2019=True)
    scenario_names = [f.name for f in scenario_folders]
    print(f"Scénarios trouvés ({len(scenario_folders)}): {scenario_names}")

    lp_values_by_scenario = []
    for scenario_folder in scenario_folders:
        lp_values = processing.process_subprocess_lp_breakdown(
            SUBPROCESS_NAME, scenario_folder, facteurs_carac_df, seuils_df,
        )
        lp_values_by_scenario.append(lp_values)
        print(f"  {scenario_folder.name}: {lp_values}")

    fig, _ = create_subprocess_contribution_chart(
        lp_values_by_scenario, scenario_names, subprocess_name=SUBPROCESS_NAME,
    )

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    output_path = config.FIGURES_DIR / "biodiversity_loss_process_contribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Sauvegardé : {output_path.name}")


if __name__ == "__main__":
    main()
