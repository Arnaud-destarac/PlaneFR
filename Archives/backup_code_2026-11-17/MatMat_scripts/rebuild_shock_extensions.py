"""
Recrée, pour un scénario de chocs (ex. .../2-shocks/1-transitions_2050/S1_2050),
les sous-dossiers extensions/dom_<ext> à partir des sorties de gmrio_to_snac_s.

Pour chaque extension <ext> :
    - detail_levels.xlsx : feuille dom_<ext> (catégories d'extension) reprise de
      <gmrio_dir>/extensions/dom_<ext>/detail_levels.xlsx, suivie des feuilles
      regions / sectors / final_demand_categories du scénario
      (<scenario_dir>/system/detail_levels.xlsx), car le système du scénario
      n'a pas la même résolution que celui de gmrio_to_snac_s ;
    - info.json : base_year / proj_year / scenario_name repris de
      <scenario_dir>/system/info.json, plus extension_name = dom_<ext>.

Le dossier <scenario_dir>/extensions est supposé avoir été vidé au préalable
(cf. run_gmrio_to_snac_s_batch.ps1).

Usage :
    python rebuild_shock_extensions.py --scenario-dir <dir> --gmrio-dir <dir>
        --extensions ext1 ext2 ...
"""

import argparse
import json
from pathlib import Path

import pandas as pd

DL_FILE = "detail_levels.xlsx"
INFO_FILE = "info.json"

# Extensions reprises des anciens accounts par update_calib_fr : leurs
# catégories ne correspondent pas à celles de gmrio_to_snac_s, on ne les
# recrée donc pas (sinon l'engine lève MEInconsistentDetailLevels).
EXCLUDED_EXTENSIONS = {"raw_materials", "ghg_emissions"}


def rebuild_extension(scenario_dir: Path, gmrio_dir: Path, extension: str):
    ext_name = f"dom_{extension}"
    gmrio_dl = gmrio_dir / "extensions" / ext_name / DL_FILE
    if not gmrio_dl.is_file():
        raise FileNotFoundError(f"Detail levels introuvables : {gmrio_dl}")

    ext_categories = pd.read_excel(gmrio_dl, sheet_name=ext_name)
    system_sheets = pd.read_excel(scenario_dir / "system" / DL_FILE, sheet_name=None)
    system_info = json.loads((scenario_dir / "system" / INFO_FILE).read_text(encoding="utf-8"))

    ext_dir = scenario_dir / "extensions" / ext_name
    ext_dir.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(ext_dir / DL_FILE, engine="openpyxl") as writer:
        ext_categories.to_excel(writer, sheet_name=ext_name, index=False)
        for sheet_name, df in system_sheets.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    info = {**system_info, "extension_name": ext_name}
    (ext_dir / INFO_FILE).write_text(json.dumps(info, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-dir", required=True, type=Path)
    parser.add_argument("--gmrio-dir", required=True, type=Path)
    parser.add_argument("--extensions", required=True, nargs="+")
    args = parser.parse_args()

    for extension in args.extensions:
        if extension in EXCLUDED_EXTENSIONS:
            print(f"  {args.scenario_dir.name} : dom_{extension} ignore")
            continue
        rebuild_extension(args.scenario_dir, args.gmrio_dir, extension)
        print(f"  {args.scenario_dir.name} : dom_{extension} recree")


if __name__ == "__main__":
    main()
