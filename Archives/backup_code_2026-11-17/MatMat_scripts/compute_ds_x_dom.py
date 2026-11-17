"""
Calcule dS, la variation relative de S (= F_x_dom / x) entre chaque scénario 2050
(S1-S4, TEND) et Base_year_2015, pour les extensions manuellement choquées
(toutes les feuilles dom_<lp> de my_manual_param.xlsx), à partir des chocs
définis dans ce fichier.

Pour un row_name et un scénario donnés :
    F_x_dom_2050 = F_x_dom_base_year * (1 + choc)   (choc appliqué à tous les secteurs)
    S_base       = F_x_dom_base_year / x_base_year
    S_2050       = F_x_dom_2050 / x_2050
    dS           = S_2050 / S_base - 1

dS est initialisé à 0 pour tous les indicateurs (même format que les feuilles Fx_*
de S_extraction.xlsx : indicateurs en ligne, secteurs en colonne) ; seules les
lignes dont le row_name figure dans my_manual_param.xlsx ET dont le choc est
renseigné et non nul sont recalculées (un choc à 0 ou vide signifie "pas encore
calibré" et laisse dS=0, sans quoi dS deviendrait le ratio compensatoire qui
annule l'effet volume de x). Les row_name de my_manual_param.xlsx absents de
S_extraction.xlsx sont ignorés.

dS_x_dom (.pkl et .xlsx) est écrit, par scénario et par extension choquée, dans
MatMat/matmat-ademe/.../2-shocks/1-transitions_2050/<scenario>/extensions/dom_<lp>/.

Les chocs "F_Y_tot" du même fichier n'ont pas de dimension sectorielle : ils sont
appliqués directement, en place, dans
Module_PlaneFR/data/<version>/[<groupe>/]<scenario>/extensions/dom_<lp>/F_Y_tot.pkl,
en recalculant à chaque fois new_value = valeur base_year (feuille FY_<lp> de
S_extraction.xlsx) x (1 + choc). Partir de la valeur base_year (et non de la
valeur déjà présente dans le F_Y_tot.pkl du scénario) rend le script idempotent
sans backup : le réexécuter ne cumule jamais le choc, même si le fichier
scénario a été régénéré entre-temps par le modèle amont.

Ces chocs sont appliqués à tous les dossiers de scénario de data/<version>, quel
que soit leur groupe (no_SSP, SSP2_2050, groupes "_agri", groupes assemblés à la
main...) : le scénario d'un dossier est lu dans son system/info.json
(scenario_name = S1...TEND, base_year = BASE_YEAR) et non déduit de son nom, de
sorte qu'un dossier renommé ou copié (ex. SSP_all/S1+) reçoit bien le choc de son
scénario.

reformat_d_cba_monde_europe.py réécrit F_Y_tot.pkl à sa valeur base_year dans tous
les dossiers de scénario : ce script doit donc être relancé après lui. L'option
--only-f-y-tot n'applique que les chocs F_Y_tot, sans recalculer les dS_x_dom
(utile en fin de chaîne quand l'engine ne sera pas relancé).
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd

MODULE_PLANEFR_DIR = Path(r"C:\Users\Arnaud\Documents\CIRED\PlaneFR\Code\Module_PlaneFR")
MATMAT_DIR = Path(r"C:\Users\Arnaud\Documents\CIRED\PlaneFR\Code\MatMat\matmat-ademe")

S_EXTRACTION_FILE = MODULE_PLANEFR_DIR / "data" / "S_extraction.xlsx"
MANUAL_PARAM_FILE = (
    MATMAT_DIR / "data" / "4-PlaneFR" / "Settings" / "a-manual_shocks" / "France" / "input" / "my_manual_param.xlsx"
)
# Version Exiobase (variable d'environnement PLANEFR_EXIOBASE_VERSION, positionnée
# par run_full_pipeline.ps1 / run_pipeline_from_engine.ps1)
EXIOBASE_VERSION = os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")
DATA_VERSION_DIR = MODULE_PLANEFR_DIR / "data" / EXIOBASE_VERSION
SHOCKS_OUTPUT_DIR = MATMAT_DIR / "data" / "4-PlaneFR" / "2-shocks" / "1-transitions_2050"

BASE_YEAR_NAME = "Base_year_2015"
# Année de base des scénarios choqués (base_year de leur system/info.json) : celle
# des valeurs FY_<lp> de S_extraction.xlsx (BASE_YEAR_NAME)
BASE_YEAR = "2015"
SCENARIO_FOLDER_NAMES = {
    "S1": "S1_2050",
    "S2": "S2_2050",
    "S3": "S3_2050",
    "S4": "S4_2050",
    "TEND": "TEND_2050",
}

# Settings du moteur Monde des scénarios SSP (run_engine_batch_SSPs.ps1)
SSP_ENGINE_SETTINGS_FILE = (
    MATMAT_DIR / "data" / "4-PlaneFR" / "Settings" / "4-engine" / "settings_engine_SSPs.json"
)


def get_ssp_base_accounts_dir():
    """Comptes Monde auxquels le moteur applique les chocs SSP : path_in.accounts de
    settings_engine_SSPs.json (relatif au dossier data de matmat-ademe).

    Source unique de l'année de base des chocs SSP : generate_shock_dS.py et
    generate_shock_agri.py y lisent la structure des chocs (S_x_dom, A,
    detail_levels, base_year), pour qu'elle soit celle des comptes réellement
    choqués par run_engine_batch_SSPs.ps1."""
    settings = json.loads(SSP_ENGINE_SETTINGS_FILE.read_text(encoding="utf-8-sig"))
    return MATMAT_DIR / "data" / settings["path_in"]["accounts"].replace("\\", "/")


def find_scenario_dirs(data_dir):
    """Dossiers de scénario (contenant extensions/) de data_dir, qu'ils soient
    directement dessous (data/<version>/<scénario>) ou dans un dossier de groupe
    (data/<version>/<groupe>/<scénario>, ex. no_SSP/S1_2050, SSP_all/S1+)."""
    found = []
    for top in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if (top / "extensions").is_dir():
            found.append(top)
            continue
        found.extend(sub for sub in sorted(top.iterdir()) if (sub / "extensions").is_dir())
    return found


def read_scenario_info(scenario_dir):
    """system/info.json d'un dossier de scénario (dict vide s'il est absent, ex.
    dossiers 2019_<région> hors France)."""
    info_path = scenario_dir / "system" / "info.json"
    if not info_path.is_file():
        return {}
    return json.loads(info_path.read_text(encoding="utf-8-sig"))


def find_shocked_scenario_dirs():
    """{scénario (S1...TEND): [dossiers de scénario]} pour tous les dossiers de
    DATA_VERSION_DIR dont le system/info.json porte ce scenario_name et l'année
    de base BASE_YEAR. Les années de base sans choc (scenario_name vide,
    Base_year_agri...) ne sont pas retenues."""
    by_scenario = {short: [] for short in SCENARIO_FOLDER_NAMES}
    for scenario_dir in find_scenario_dirs(DATA_VERSION_DIR):
        info = read_scenario_info(scenario_dir)
        if info.get("scenario_name") in by_scenario and str(info.get("base_year")) == BASE_YEAR:
            by_scenario[info["scenario_name"]].append(scenario_dir)
    return by_scenario


def get_manual_lp_names():
    """Sous-processus (lp) ayant une feuille dom_<lp> dans my_manual_param.xlsx."""
    sheet_names = pd.ExcelFile(MANUAL_PARAM_FILE).sheet_names
    return [name[len("dom_"):] for name in sheet_names if name.startswith("dom_")]


def load_x_domestic():
    """Feuille 'x' de S_extraction.xlsx, restreinte aux secteurs domestiques
    (mêmes colonnes region/category/sub_category/sector que F_x_dom)."""
    x = pd.read_excel(S_EXTRACTION_FILE, sheet_name="x", header=list(range(5)), index_col=0)
    return x.xs("domestic", axis=1, level="origin")


def load_f_x_dom_base(lp_name):
    return pd.read_excel(S_EXTRACTION_FILE, sheet_name=f"Fx_{lp_name}", header=list(range(4)), index_col=0)


def load_f_y_tot_base(lp_name):
    return pd.read_excel(S_EXTRACTION_FILE, sheet_name=f"FY_{lp_name}", header=list(range(2)), index_col=0)


def load_manual_shocks(lp_name):
    """Reconstruit un tableau propre à partir de my_manual_param.xlsx : la ligne 1
    porte les noms de scénario (S1..TEND), la ligne 3 les noms de champ
    (variable, row_region, row_name, ...), les données commencent ligne 4."""
    raw = pd.read_excel(MANUAL_PARAM_FILE, sheet_name=f"dom_{lp_name}", header=None)
    scenario_cols = raw.iloc[0, 7:].tolist()
    field_cols = raw.iloc[2, 0:7].tolist()
    data = raw.iloc[3:].reset_index(drop=True)
    data.columns = field_cols + scenario_cols
    return data


def compute_ds(lp_name, x_dom):
    f_base = load_f_x_dom_base(lp_name)
    ds_by_scenario = {
        short: pd.DataFrame(0.0, index=f_base.index, columns=f_base.columns) for short in SCENARIO_FOLDER_NAMES
    }

    shocks = load_manual_shocks(lp_name)
    x_shocks = shocks[shocks["variable"] == "F_x_dom"]

    for _, row in x_shocks.iterrows():
        row_name = row["row_name"]
        if row_name not in f_base.index:
            print(f"[skip] {lp_name}: row_name '{row_name}' absent de Fx_{lp_name} (S_extraction.xlsx), ignoré.")
            continue

        f_row_base = f_base.loc[row_name]
        s_base = f_row_base / x_dom.loc[BASE_YEAR_NAME]

        for short, folder_name in SCENARIO_FOLDER_NAMES.items():
            shock = row[short]
            if pd.isna(shock) or shock == 0:
                continue
            f_row_2050 = f_row_base * (1 + shock)
            s_2050 = f_row_2050 / x_dom.loc[folder_name]
            ds_row = (s_2050 / s_base - 1).replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
            ds_by_scenario[short].loc[row_name] = ds_row

    return ds_by_scenario


def make_pickle_portable(df):
    """Force les niveaux d'index/colonnes et les colonnes de type texte en
    dtype 'object' avant pickling. Le format pickle des dtypes 'string'
    (StringDtype, y compris backend pyarrow) n'est pas stable d'une version
    de pandas à l'autre : un pickle écrit avec pandas >= 3.0 (où le texte
    est en StringDtype par défaut) ne peut pas être relu par le pandas 2.x
    utilisé par matmat-ademe (.venv), quel que soit l'interpréteur Python
    utilisé pour lancer ce script."""

    def sanitize_index(index):
        if isinstance(index, pd.MultiIndex):
            if any(isinstance(level.dtype, pd.StringDtype) for level in index.levels):
                return index.set_levels(
                    [
                        level.astype(object) if isinstance(level.dtype, pd.StringDtype) else level
                        for level in index.levels
                    ]
                )
            return index
        if isinstance(index.dtype, pd.StringDtype):
            return index.astype(object)
        return index

    df = df.copy()
    df.index = sanitize_index(df.index)
    df.columns = sanitize_index(df.columns)
    string_cols = [c for c in df.columns if isinstance(df[c].dtype, pd.StringDtype)]
    if string_cols:
        df[string_cols] = df[string_cols].astype(object)
    return df


def write_ds_outputs(lp_name, ds_by_scenario):
    for short, folder_name in SCENARIO_FOLDER_NAMES.items():
        ds = ds_by_scenario[short]
        if (ds == 0).all().all():
            continue  # aucun choc F_x_dom pour ce couple scénario/extension

        out_dir = SHOCKS_OUTPUT_DIR / folder_name / "extensions" / f"dom_{lp_name}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if not isinstance(ds.index, pd.MultiIndex):
            ds.index = pd.MultiIndex.from_arrays([ds.index], names=ds.index.names)
        make_pickle_portable(ds).to_pickle(out_dir / "dS_x_dom.pkl")
        ds.to_excel(out_dir / "dS_x_dom.xlsx", sheet_name="dS_x_dom")
        print(f"[write] {out_dir / 'dS_x_dom.xlsx'} (+ .pkl)")


def apply_f_y_tot_shocks(lp_name, scenario_dirs):
    """Applique les chocs F_Y_tot de l'extension à tous les dossiers de scénario.

    scenario_dirs : {scénario (S1...TEND): [dossiers]}, cf. find_shocked_scenario_dirs.
    """
    shocks = load_manual_shocks(lp_name)
    y_shocks = shocks[shocks["variable"] == "F_Y_tot"]
    if y_shocks.empty:
        return
    fy_base = load_f_y_tot_base(lp_name)

    for _, row in y_shocks.iterrows():
        row_name = row["row_name"]
        if row_name not in fy_base.index:
            print(f"[skip] {lp_name}: row_name '{row_name}' absent de FY_{lp_name} (S_extraction.xlsx), ignoré.")
            continue
        base_value = fy_base.loc[row_name]

        for short in SCENARIO_FOLDER_NAMES:
            shock = row[short]
            if pd.isna(shock) or shock == 0:
                continue

            for scenario_dir in scenario_dirs[short]:
                label = scenario_dir.relative_to(DATA_VERSION_DIR).as_posix()
                f_y_path = scenario_dir / "extensions" / f"dom_{lp_name}" / "F_Y_tot.pkl"
                if not f_y_path.exists():
                    # Le F_Y_tot.pkl du scénario est créé par reformat_d_cba_monde_europe
                    # dans les dossiers produits par l'engine, qui a lui-même besoin des
                    # dS_x_dom écrits plus haut : on ne bloque pas le calcul des dS, il
                    # suffira de relancer ce script après engine + reformat.
                    print(f"[skip] {label}/dom_{lp_name}: F_Y_tot.pkl absent (engine/reformat_d_cba_monde_europe pas encore lancés ?), choc '{row_name}' non appliqué.")
                    continue
                f_y = pd.read_pickle(f_y_path)
                if row_name not in f_y.index:
                    print(f"[skip] {label}/dom_{lp_name}: '{row_name}' absent de F_Y_tot.pkl, ignoré.")
                    continue

                f_y.loc[row_name] = base_value * (1 + shock)
                f_y.to_pickle(f_y_path)
                print(f"[update] {label}/dom_{lp_name}/F_Y_tot.pkl : '{row_name}' = base_year x (1{shock:+.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les dS_x_dom des chocs manuels et applique les chocs F_Y_tot."
    )
    parser.add_argument(
        "--only-f-y-tot",
        action="store_true",
        help="n'applique que les chocs F_Y_tot, sans recalculer ni réécrire les dS_x_dom",
    )
    args = parser.parse_args()

    scenario_dirs = find_shocked_scenario_dirs()
    for short, dirs in scenario_dirs.items():
        names = ", ".join(d.relative_to(DATA_VERSION_DIR).as_posix() for d in dirs) or "aucun"
        print(f"[scenarios] {short} : {len(dirs)} dossier(s) ({names})")

    x_dom = None if args.only_f_y_tot else load_x_domestic()

    for lp_name in get_manual_lp_names():
        print(f"=== {lp_name} ===")
        if not args.only_f_y_tot:
            ds_by_scenario = compute_ds(lp_name, x_dom)
            write_ds_outputs(lp_name, ds_by_scenario)
        apply_f_y_tot_shocks(lp_name, scenario_dirs)


if __name__ == "__main__":
    main()
