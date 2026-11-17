"""
Génère, pour chaque jeu de chocs de data/4-PlaneFR/2-shocks (1-transitions_2050,
3-SSPs_2050), une variante '_agri' (1-transitions_2050_agri, 3-SSPs_2050_agri)
dans laquelle seuls les stressors des secteurs 'Food' subsistent :

    - colonnes (region, ..., sector) des secteurs Food : dS = dS du scénario
      (0 si le scénario ne choque pas l'extension) ;
    - toutes les autres colonnes : dS = -1, soit S_2050 = S_base x (1 + dS) = 0.

Exceptions pour les extensions en S_Z (cf. S_Z_FILTERS) : dom_ghg_combustion
garde les colonnes Food + engrais, dom_raw_materials garde les lignes Food
(produits prélevés) plutôt que les colonnes. Les extensions de GES en S_x_dom
(dom_ghg_emissions, ghg_emissions) gardent aussi les colonnes des engrais
(cf. S_X_DOM_EXTRA_SECTORS).

Les secteurs Food sont les 'sector' ayant un 1 dans la colonne 'Food' de la
feuille 'M' de Module_PlaneFR/data/bridge_matrices.xlsx, auxquels on ajoute
FOOD_EXTRA_SECTORS et dont on retire FOOD_EXCLUDED_SECTORS.

Chaque scénario <scenario> est d'abord copié tel quel, sans les .xlsx de
données (system/ : dA, dY, dY_k en .pkl, detail_levels, info ; extensions/ :
detail_levels, info), puis les dS (.pkl uniquement) de toutes les extensions
sont réécrits. Les extensions traitées sont toutes celles des accounts de base
(EXTENSIONS_BASE_DIRS) ayant un S_x_dom.pkl (choc dS_x_dom) ou, à défaut, un
S_Z.pkl (choc dS_Z, colonnes = secteurs utilisateurs ; S_Y n'est pas choqué) ;
celles absentes du dossier de choc d'origine sont créées (detail_levels.xlsx
copié depuis les accounts de base, info.json).

Un scénario supplémentaire 'Base_year_agri' est créé dans chaque dossier
'_agri' : aucun choc sur le système (system/ ne contient que detail_levels.xlsx
et info.json), dS = 0 sur les secteurs Food (S inchangé) et -1 ailleurs.
"""

import json
import shutil

import pandas as pd

from compute_ds_x_dom import MATMAT_DIR, MODULE_PLANEFR_DIR, get_ssp_base_accounts_dir, make_pickle_portable

SHOCKS_DIR = MATMAT_DIR / "data" / "4-PlaneFR" / "2-shocks"
OUTPUTS_DIR = MATMAT_DIR / "data" / "4-PlaneFR" / "Outputs"
BRIDGE_FILE = MODULE_PLANEFR_DIR / "data" / "bridge_matrices.xlsx"

AGRI_SUFFIX = "_agri"
BASE_YEAR_SCENARIO = "Base_year_agri"

# Jeu de chocs -> dossier extensions des accounts de base correspondants. Pour les
# chocs SSP, ce sont les comptes Monde auxquels le moteur les applique
# (path_in.accounts de settings_engine_SSPs.json), comme dans generate_shock_dS.py.
EXTENSIONS_BASE_DIRS = {
    "1-transitions_2050": OUTPUTS_DIR / "France" / "Base_year_2015" / "extensions",
    "3-SSPs_2050": get_ssp_base_accounts_dir() / "extensions",
}

# Coefficients d'extension choqués, par ordre de préférence
S_NAMES = ("S_x_dom", "S_Z")

FOOD_EXTRA_SECTORS = {"Products of forestry, logging and related services (02)"}
FOOD_EXCLUDED_SECTORS = {"Hotel and restaurant services (55)"}
FERTILIZER_SECTORS = {"N-fertiliser", "P- and other fertiliser"}

# Extensions en S_Z (lignes = produits, colonnes = secteurs utilisateurs) :
# axe filtré et secteurs supplémentaires conservés. Par défaut (S_x_dom), on
# garde les colonnes des secteurs Food.
#   - ghg_combustion : colonnes Food + engrais -> émissions de combustion des
#     secteurs de l'agri-food system ;
#   - raw_materials : lignes Food -> toute la biomasse prélevée (agricole et
#     forestière), quel que soit le secteur qui l'utilise.
S_Z_FILTERS = {
    "dom_ghg_combustion": ("columns", FERTILIZER_SECTORS),
    "dom_raw_materials": ("index", set()),
}

# Extensions en S_x_dom pour lesquelles les engrais sont conservés en plus des
# secteurs Food (émissions de GES attribuables à l'agri-food system).
S_X_DOM_EXTRA_SECTORS = {
    "dom_ghg_emissions": FERTILIZER_SECTORS,
    "ghg_emissions": FERTILIZER_SECTORS,
}


def load_food_sectors():
    bridge = pd.read_excel(BRIDGE_FILE, sheet_name="M")
    food = set(bridge.loc[bridge["Food"] == 1, "sector"])
    return (food | FOOD_EXTRA_SECTORS) - FOOD_EXCLUDED_SECTORS


def get_filter(ext_name, s_name, food_sectors):
    """(axe, secteurs conservés) pour une extension : S_Z_FILTERS pour les
    extensions en S_Z, sinon colonnes Food (+ S_X_DOM_EXTRA_SECTORS)."""
    if s_name == "S_Z":
        if ext_name not in S_Z_FILTERS:
            raise ValueError(f"Extension en S_Z sans filtre défini dans S_Z_FILTERS : {ext_name}")
        axis, extra = S_Z_FILTERS[ext_name]
        return axis, food_sectors | extra
    return "columns", food_sectors | S_X_DOM_EXTRA_SECTORS.get(ext_name, set())


def get_extensions(group):
    """Extension -> coefficients choqués : S_x_dom si présent, sinon S_Z
    (extensions en format produit, ex. dom_raw_materials des accounts France)."""
    extensions = {}
    for ext in sorted(EXTENSIONS_BASE_DIRS[group].iterdir()):
        for s_name in S_NAMES:
            if (ext / f"{s_name}.pkl").is_file():
                extensions[ext.name] = s_name
                break
    return extensions


def compute_ds_agri(ds, axis, kept_sectors):
    """Secteurs conservés (sur l'axe donné, "columns" ou "index") : dS inchangé ;
    autres secteurs : dS = -1 (S = 0)."""
    ds = ds.copy()
    if axis == "index":
        is_kept = ds.index.get_level_values("sector").isin(kept_sectors)
        ds.loc[~is_kept, :] = -1.0
    else:
        is_kept = ds.columns.get_level_values("sector").isin(kept_sectors)
        ds.loc[:, ~is_kept] = -1.0
    return ds


def write_ds(out_dir, ds, ds_name):
    if not isinstance(ds.index, pd.MultiIndex):
        ds.index = pd.MultiIndex.from_arrays([ds.index], names=ds.index.names)
    make_pickle_portable(ds).to_pickle(out_dir / f"{ds_name}.pkl")
    print(f"[write] {out_dir / f'{ds_name}.pkl'}")


def write_extension_agri(group, scenario_dir, ext_name, s_name, info, food_sectors):
    """Réécrit d<s_name> (dS_x_dom ou dS_Z) de l'extension dans scenario_dir
    (dossier _agri), à partir du dS déjà présent (copié du scénario d'origine)
    ou de 0 s'il n'y en a pas."""
    base_ext_dir = EXTENSIONS_BASE_DIRS[group] / ext_name
    out_dir = scenario_dir / "extensions" / ext_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (out_dir / "detail_levels.xlsx").exists():
        shutil.copy2(base_ext_dir / "detail_levels.xlsx", out_dir / "detail_levels.xlsx")
    if not (out_dir / "info.json").exists():
        ext_info = {**info, "extension_name": ext_name}
        (out_dir / "info.json").write_text(json.dumps(ext_info, indent=2), encoding="utf-8")

    ds_name = f"d{s_name}"
    ds_path = out_dir / f"{ds_name}.pkl"
    if ds_path.exists():
        ds = pd.read_pickle(ds_path)
    else:
        s_base = pd.read_pickle(base_ext_dir / f"{s_name}.pkl")
        ds = pd.DataFrame(0.0, index=s_base.index, columns=s_base.columns)
    axis, kept_sectors = get_filter(ext_name, s_name, food_sectors)
    write_ds(out_dir, compute_ds_agri(ds, axis, kept_sectors), ds_name)


def check_food_sectors(group, ext_name, s_name, food_sectors):
    s_base = pd.read_pickle(EXTENSIONS_BASE_DIRS[group] / ext_name / f"{s_name}.pkl")
    axis, kept_sectors = get_filter(ext_name, s_name, food_sectors)
    labels = s_base.index if axis == "index" else s_base.columns
    unknown = sorted(kept_sectors - set(labels.get_level_values("sector")))
    if unknown:
        print(f"[warn] {group} / {ext_name} : secteurs conservés absents de {s_name} ({axis}) : {unknown}")


def ignore_data_xlsx(_dir, names):
    """Seuls les .pkl sont copiés pour les données ; .xlsx gardé pour detail_levels."""
    return [n for n in names if n.endswith(".xlsx") and n != "detail_levels.xlsx"]


def make_scenario_agri(group, src_dir, out_dir, extensions, food_sectors):
    print(f"=== {out_dir.parent.name} / {out_dir.name} ===")
    shutil.copytree(src_dir, out_dir, ignore=ignore_data_xlsx)
    info = json.loads((out_dir / "system" / "info.json").read_text(encoding="utf-8"))
    for ext_name, s_name in extensions.items():
        write_extension_agri(group, out_dir, ext_name, s_name, info, food_sectors)
    return info


def make_base_year_agri(group, template_dir, out_dir, extensions, food_sectors):
    """Base_year_agri : system sans choc, dS = 0 (Food) / -1 (autres)."""
    print(f"=== {out_dir.parent.name} / {out_dir.name} ===")
    template_info = json.loads((template_dir / "system" / "info.json").read_text(encoding="utf-8"))
    base_year = template_info["base_year"]
    info = {"base_year": base_year, "proj_year": base_year, "scenario_name": BASE_YEAR_SCENARIO}

    system_dir = out_dir / "system"
    system_dir.mkdir(parents=True)
    shutil.copy2(template_dir / "system" / "detail_levels.xlsx", system_dir / "detail_levels.xlsx")
    (system_dir / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    for ext_name, s_name in extensions.items():
        ext_dir = out_dir / "extensions" / ext_name
        ext_dir.mkdir(parents=True)
        # detail_levels repris du scénario d'origine (même résolution que les chocs)
        template_dl = template_dir / "extensions" / ext_name / "detail_levels.xlsx"
        if template_dl.exists():
            shutil.copy2(template_dl, ext_dir / "detail_levels.xlsx")
        write_extension_agri(group, out_dir, ext_name, s_name, info, food_sectors)


def main():
    food_sectors = load_food_sectors()
    print(f"{len(food_sectors)} secteurs Food : {sorted(food_sectors)}")

    for group in EXTENSIONS_BASE_DIRS:
        src_group_dir = SHOCKS_DIR / group
        out_group_dir = SHOCKS_DIR / f"{group}{AGRI_SUFFIX}"
        if out_group_dir.exists():
            shutil.rmtree(out_group_dir)
        out_group_dir.mkdir(parents=True)
        if (src_group_dir / "metadata.json").exists():
            shutil.copy2(src_group_dir / "metadata.json", out_group_dir / "metadata.json")

        extensions = get_extensions(group)
        for ext_name, s_name in extensions.items():
            check_food_sectors(group, ext_name, s_name, food_sectors)

        scenario_dirs = sorted(d for d in src_group_dir.iterdir() if d.is_dir())
        for src_dir in scenario_dirs:
            make_scenario_agri(group, src_dir, out_group_dir / f"{src_dir.name}{AGRI_SUFFIX}", extensions, food_sectors)

        make_base_year_agri(group, scenario_dirs[0], out_group_dir / BASE_YEAR_SCENARIO, extensions, food_sectors)


if __name__ == "__main__":
    main()
