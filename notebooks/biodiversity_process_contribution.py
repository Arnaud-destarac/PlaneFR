"""
Contribution of the different Earth-system processes (LP, "EXIOBASE groups"
column of characterization_factors.xlsx) to the footprint of the
"Biodiversity loss" sub-process, in % of the total footprint -- one 100 %
stacked bar per France scenario (base scenario + 2050 transition scenarios,
see config.get_scenario_folders).

ghg_combustion and ghg_emissions are grouped under "GHG emissions"; the other
LPs are renamed for display (see planefr_lib.plot_process_breakdown).

Generates figures/biodiversity_loss_process_contribution.png.
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

SUBPROCESS_NAME = "Biodiversity loss - PDF"


def main():
    cf_df = io.load_characterization_factors(config.SCENARIO_GROUP)
    thresholds_df = io.load_thresholds(config.SCENARIO_GROUP)

    scenario_folders = config.get_scenario_folders(exclude_2019=True)
    scenario_names = [f.name for f in scenario_folders]
    print(f"Scenarios found ({len(scenario_folders)}): {scenario_names}")

    lp_values_by_scenario = []
    for scenario_folder in scenario_folders:
        lp_values = processing.process_subprocess_lp_breakdown(
            SUBPROCESS_NAME, scenario_folder, cf_df, thresholds_df,
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
    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    main()
