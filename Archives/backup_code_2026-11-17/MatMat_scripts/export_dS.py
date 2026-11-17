"""
Extrait les données de Base_year_2015 (system/x.pkl + extensions dom_*/F_x_dom.pkl
et dom_*/F_Y_tot.pkl) vers un classeur Excel, une feuille par tableau.

Feuilles produites :
- "x" : x.pkl de Base_year_2015, transposé (secteurs en colonnes, comme F_x_dom),
  avec une ligne supplémentaire par scénario 2050 (S1-S4, TEND) pour comparaison.
  La colonne "scenario" indique le nom du dossier d'origine de chaque ligne.
- "Fx_<lp>" : F_x_dom.pkl de chaque extension dom_* de Base_year_2015
  (indicateurs en ligne, secteurs en colonne, format d'origine).
- "FY_<lp>" : F_Y_tot.pkl de chaque extension dom_* de Base_year_2015, quand ce
  fichier existe (absent pour ghg_combustion, ghg_emissions, raw_materials).
"""

import sys
from pathlib import Path

import pandas as pd

MODULE_PLANEFR_DIR = Path(r"C:\Users\Arnaud\Documents\CIRED\PlaneFR\Code\Module_PlaneFR")

sys.path.insert(0, str(MODULE_PLANEFR_DIR / "notebooks"))
from planefr_lib import config  # noqa: E402

BASE_YEAR_FOLDER = config.BASE_DATA_DIR / "no_SSP" / "Base_year_2015"
OTHER_SCENARIO_FOLDERS = [
    d for d in config.get_scenario_folders(exclude_2019=True) if d.name != BASE_YEAR_FOLDER.name
]

OUTPUT_FILE = config.PROJECT_DIR / "data" / "S_extraction.xlsx"


def load_x_row(scenario_folder):
    """Charge system/x.pkl et le transpose en une ligne (secteurs en colonnes)."""
    x = pd.read_pickle(scenario_folder / "system" / "x.pkl")
    return x.iloc[:, 0]  # Series indexée par (origin, region, category, sub_category, sector)


def build_x_sheet():
    scenario_folders = [BASE_YEAR_FOLDER] + OTHER_SCENARIO_FOLDERS
    rows = [load_x_row(folder) for folder in scenario_folders]
    df = pd.concat(rows, axis=1).T
    df.index = pd.Index([f.name for f in scenario_folders], name="scenario")
    return df


def get_dom_extension_names(scenario_folder):
    """Noms des sous-processus (lp) ayant un dossier d'extensions dom_<lp>."""
    ext_dir = scenario_folder / "extensions"
    return sorted(
        p.name[len("dom_"):]
        for p in ext_dir.iterdir()
        if p.is_dir() and p.name.startswith("dom_")
    )


def sheet_name(prefix, lp_name):
    return f"{prefix}_{lp_name}"[:31]


def main():
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        build_x_sheet().to_excel(writer, sheet_name="x")

        for lp_name in get_dom_extension_names(BASE_YEAR_FOLDER):
            ext_dir = BASE_YEAR_FOLDER / "extensions" / f"dom_{lp_name}"

            f_x_dom_path = ext_dir / "F_x_dom.pkl"
            if f_x_dom_path.exists():
                pd.read_pickle(f_x_dom_path).to_excel(writer, sheet_name=sheet_name("Fx", lp_name))
            else:
                print(f"[info] F_x_dom.pkl absent pour {lp_name}, feuille non créée.")

            f_y_tot_path = ext_dir / "F_Y_tot.pkl"
            if f_y_tot_path.exists():
                pd.read_pickle(f_y_tot_path).to_excel(writer, sheet_name=sheet_name("FY", lp_name))
            else:
                print(f"[info] F_Y_tot.pkl absent pour {lp_name}, feuille non créée.")

    print(f"Classeur écrit : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
