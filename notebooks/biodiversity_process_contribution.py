"""
Contribution of the different Earth-system processes (LP, "EXIOBASE groups"
column of indicator_computing.xlsx) to the footprint of the
"Biodiversity loss" indicator, in % of the total footprint -- one 100 %
stacked bar per France scenario (base scenario + 2050 transition scenarios,
see config.get_scenario_folders).

ghg_combustion and ghg_emissions are grouped under "GHG emissions"; the other
LPs are renamed for display (see planefr_lib.plot_process_breakdown).

Generates figures/biodiversity_process_contribution/biodiversity_loss_process_contribution.png
and the Excel workbook of its data (.xlsx, same name).
"""

import sys
from pathlib import Path

notebooks_dir = Path(__file__).resolve().parent
while not (notebooks_dir / "planefr_lib").exists() and notebooks_dir != notebooks_dir.parent:
    notebooks_dir = notebooks_dir.parent
if str(notebooks_dir) not in sys.path:
    sys.path.insert(0, str(notebooks_dir))

import matplotlib.pyplot as plt

from planefr_lib import config, figure_export, io, processing
from planefr_lib.plot_process_breakdown import create_biodiversity_process_contribution

# Folder of the figures: figures/<NOTEBOOK_NAME>/
NOTEBOOK_NAME = "biodiversity_process_contribution"
INDICATOR_NAME = "Biodiversity loss - PDF"


def main():
    cf_df = io.load_characterization_factors(config.SCENARIO_GROUP)
    boundaries_df = io.load_boundaries(config.SCENARIO_GROUP)

    scenario_folders = config.get_scenario_folders(exclude_2019=True)
    scenario_names = [f.name for f in scenario_folders]
    print(f"Scenarios found ({len(scenario_folders)}): {scenario_names}")

    lp_values_by_scenario = []
    for scenario_folder in scenario_folders:
        lp_values = processing.process_indicator_lp_breakdown(
            INDICATOR_NAME, scenario_folder, cf_df, boundaries_df,
        )
        lp_values_by_scenario.append(lp_values)
        print(f"  {scenario_folder.name}: {lp_values}")

    output_path = config.figure_dir(NOTEBOOK_NAME) / "biodiversity_loss_process_contribution.png"
    fig, _ = create_biodiversity_process_contribution(
        lp_values_by_scenario, scenario_names, indicator_name=INDICATOR_NAME,
        unit=figure_export.indicator_unit(boundaries_df, INDICATOR_NAME),
        excel_path=output_path.with_suffix(".xlsx"),
    )
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    main()
