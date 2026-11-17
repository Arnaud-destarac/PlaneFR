"""
Pour chaque extension dans Outputs/World/Base_year_2015 et Base_year_2019/extensions :
- transforme d_cba.pkl / F_x_dom.pkl / F_Y_tot.pkl en versions Monde, Europe et une par région Exiobase
- Monde  : supprime les Y_category indésirables, somme tout -> (region="World", Y_category="all", sector="all")
- Europe : idem mais en ne gardant que les 27 premières régions -> (region="Europe", ...)
- Région (hors FR) : idem mais en ne gardant qu'une seule région Exiobase (ex. "DK"...)
  -> (region=<code>, ...) ; mêmes fichiers (d_cba/F_x_dom/F_Y_tot) et même arborescence
  que Monde/Europe (dossier <ext_name>, pas de préfixe dom_), exportée dans 2019_<code>
  (une par région listée dans detail_levels.xlsx, UE et hors UE)
- France (FR) : cas particulier conservé tel quel -- seul F_Y_tot.pkl est exporté,
  dans dom_{ext_name}, car d_cba/F_x_dom pour la France proviennent déjà d'un autre
  pipeline (scénarios France dom_/imp_ existants dans data/<version>)

Le F_Y_tot France de chaque année de base (SRC_DIR / SRC_BASE_YEARS) est copié dans
dom_{ext} de tous les dossiers de scénario France de MÊME année de base déjà
présents dans data/<version>, qu'ils soient directement dessous (ex. 2019_FR) ou
dans un dossier de groupe (data/<version>/<groupe>/<scénario>, ex. no_SSP/S1_2050,
SSP2_2050/TEND_2050, Base_year/2019_FR). L'année de base d'un dossier est lue dans
son system/info.json (base_year) et non déduite de son emplacement : une copie de
2019_FR placée dans un groupe reçoit bien le F_Y_tot 2019, et les scénarios 2050
(année de base 2015) le F_Y_tot 2015.

Base_year_2019 (SRC_DIR[1]) produit en plus la version Monde, Europe, et une version
par région Exiobase hors France, chacune dans son dossier dédié (2019_W, 2019_EU27,
2019_<code>).

Ce script réécrit F_Y_tot.pkl à sa valeur de l'année de base : compute_ds_x_dom.py
(chocs F_Y_tot des scénarios) doit être relancé après lui.
"""

import json
import os
import pickle
import shutil
import pandas as pd

# Chemins (relatifs à ce script, pas au répertoire de travail courant)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PLANEFR_DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "4-PlaneFR")

SRC_DIR = [
    os.path.join(PLANEFR_DATA_DIR, "Outputs", "World", "Base_year_2015", "extensions"),
    os.path.join(PLANEFR_DATA_DIR, "Outputs", "World", "Base_year_2019", "extensions"),
]
# Année de base des comptes de chaque dossier de SRC_DIR (même ordre)
SRC_BASE_YEARS = [2015, 2019]

# Version Exiobase (variable d'environnement PLANEFR_EXIOBASE_VERSION, positionnée
# par run_full_pipeline.ps1 / run_pipeline_from_engine.ps1)
EXIOBASE_VERSION = os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")
DATA_DIR = os.path.join("C:\\Users\\Arnaud\\Documents\\CIRED\\PlaneFR\\Code\\Module_PlaneFR\\data", EXIOBASE_VERSION)
DST_MONDE = os.path.join(DATA_DIR, "2019_W", "extensions")
DST_EUROPE = os.path.join(DATA_DIR, "2019_EU27", "extensions")

# Y_categories à exclure. Ce label ("Exports: Total (fob)") est celui des données
# BRUTES en amont (SRC_DIR) ; il ne faut pas le confondre avec le label "Exports"
# utilisé par les notebooks d'analyse (planefr_lib.io.exclude_y_category), qui
# lisent eux les données déjà reformatées dans data/<version> — deux étages du
# pipeline, deux libellés différents pour le même concept, pas une incohérence.
YCATS_EXCLUDED = {"Exports: Total (fob)"}

# Liste de toutes les régions Exiobase (UE puis hors UE / agrégats "Rest of World")
# issue du fichier de référence ; les 27 premières sont les régions européennes (UE).
DETAIL_LEVELS_PATH = os.path.join(PLANEFR_DATA_DIR, "Outputs", "World", "Base_year_2019", "system", "detail_levels.xlsx")
with open(DETAIL_LEVELS_PATH, "rb") as f:
    _regions_df = pd.read_excel(f, sheet_name="regions")
ALL_REGIONS = _regions_df["region"].astype(str).to_list()
EU_REGIONS = ALL_REGIONS[:27]

# Région traitée à part : seul F_Y_tot.pkl est exporté, dans les dossiers de
# scénario France existants (cf. FRANCE_DIRS)
FRANCE_CODE = "FR"
# Extensions dont le F_Y France vient de la calibration France (pas de F_Y_tot exporté)
FRANCE_EXCLUDED_EXTENSIONS = {"ghg_emissions", "raw_materials"}

# Une destination 2019_<code>/extensions par région Exiobase hors France (ex. 2019_DK...)
REGION_TARGETS = [
    (code, os.path.join(DATA_DIR, f"2019_{code}", "extensions")) for code in ALL_REGIONS if code != FRANCE_CODE
]

# Dossiers de sortie entièrement produits par ce script (Monde, Europe, régions hors
# France) : jamais des scénarios France.
EXCLUDED_DIRS = {"2019_W", "2019_EU27"} | {f"2019_{code}" for code in ALL_REGIONS if code != FRANCE_CODE}


def _find_scenario_extension_dirs(data_dir: str) -> list[str]:
    """Liste les dossiers <scénario>/extensions et <groupe>/<scénario>/extensions
    de data_dir, hors dossiers de sortie de ce script (EXCLUDED_DIRS). Deux
    arborescences coexistent :
    - data/<version>/<scénario>/extensions            (ex. 2019_FR)
    - data/<version>/<groupe>/<scénario>/extensions   (ex. no_SSP/S1_2050, Base_year/2019_FR)
    """
    found = []
    for d in sorted(os.listdir(data_dir)):
        top = os.path.join(data_dir, d)
        if d in EXCLUDED_DIRS or not os.path.isdir(top):
            continue
        if os.path.isdir(os.path.join(top, "extensions")):
            found.append(os.path.join(top, "extensions"))
            continue
        for sub in sorted(os.listdir(top)):
            if os.path.isdir(os.path.join(top, sub, "extensions")):
                found.append(os.path.join(top, sub, "extensions"))
    return found


def _is_france_scenario(extensions_dir: str) -> bool:
    """True si les extensions sont séparées en dom_<ext> / imp_<ext> (sorties du
    moteur France), contrairement aux dossiers Monde/Europe/région (un seul
    dossier <ext>), y compris une éventuelle copie de ceux-ci dans un groupe."""
    return any(d.startswith(("dom_", "imp_")) for d in os.listdir(extensions_dir))


def _scenario_base_year(extensions_dir: str) -> int:
    """Année de base d'un scénario France : base_year de son system/info.json. À
    défaut (fichier absent ou incomplet), 2019 si le nom du dossier contient
    "2019", 2015 sinon."""
    scenario_dir = os.path.dirname(extensions_dir)
    try:
        with open(os.path.join(scenario_dir, "system", "info.json"), encoding="utf-8-sig") as f:
            return int(json.load(f)["base_year"])
    except (OSError, KeyError, TypeError, ValueError):
        year = 2019 if "2019" in os.path.basename(scenario_dir) else 2015
        print(f"[warn] {scenario_dir} : base_year illisible dans system/info.json, {year} déduit du nom du dossier")
        return year


def _find_france_dirs_by_base_year(data_dir: str) -> dict[int, list[str]]:
    """{année de base: [dossiers extensions/ des scénarios France de cette année]},
    pour les années de SRC_BASE_YEARS."""
    by_year = {year: [] for year in SRC_BASE_YEARS}
    for extensions_dir in _find_scenario_extension_dirs(data_dir):
        if not _is_france_scenario(extensions_dir):
            continue
        year = _scenario_base_year(extensions_dir)
        if year in by_year:
            by_year[year].append(extensions_dir)
        else:
            print(f"[warn] {extensions_dir} : année de base {year} sans comptes Monde (SRC_BASE_YEARS), F_Y_tot non exporté")
    return by_year


# Dossiers de scénario France déjà présents dans data/<version>, par année de base
FRANCE_DIRS = _find_france_dirs_by_base_year(DATA_DIR)


def transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode='monde'  -> toutes les régions sauf Y_cats exclus, somme -> World/all/all
    mode='europe' -> 27 premières régions sauf Y_cats exclus, somme -> Europe/all/all
    mode=<code>   -> une seule région Exiobase (ex. 'FR', 'DK'...), somme -> <code>/all/all
    """
    if "Y_category" in df.columns.names:
        mask_ycat = ~df.columns.get_level_values("Y_category").isin(YCATS_EXCLUDED)
        df = df.loc[:, mask_ycat]

    if mode == "europe":
        df = df.loc[:, df.columns.get_level_values("region").isin(EU_REGIONS)]
    elif mode != "monde":
        df = df.loc[:, df.columns.get_level_values("region") == mode]

    totals = df.sum(axis=1)

    region_label = {"monde": "World", "europe": "Europe"}.get(mode, mode)
    if "Y_category" in df.columns.names:
        col_index = pd.MultiIndex.from_tuples([(region_label, "all", "all")], names=["region", "Y_category", "sector"])
    else:
        col_index = pd.MultiIndex.from_tuples([(region_label, "all")], names=["region", "sector"])

    return pd.DataFrame(totals.values, index=df.index, columns=col_index)


def _save_result(out_dir, filename, result, label):
    """Écrit `result` en pickle sous out_dir/filename (en créant les dossiers manquants), avec un log."""
    os.makedirs(out_dir, exist_ok=True)
    out_pkl = os.path.join(out_dir, filename)
    with open(out_pkl, "wb") as f:
        pickle.dump(result, f)
    print(f"  [{label.upper()}] {out_pkl}  shape={result.shape}")


def process_extension(ext_name: str, src_dir: str):
    base_year = SRC_BASE_YEARS[SRC_DIR.index(src_dir)]
    srcs_pkl = [
        os.path.join(src_dir, ext_name, "d_cba.pkl"),
        os.path.join(src_dir, ext_name, "F_x_dom.pkl"),
        os.path.join(src_dir, ext_name, "F_Y_tot.pkl"),
    ]

    for src_pkl in srcs_pkl:
        if not os.path.exists(src_pkl):
            print(f"  [SKIP] {os.path.basename(src_pkl)} introuvable dans {ext_name}")
            continue

        with open(src_pkl, "rb") as f:
            df = pickle.load(f)

        # Certaines extensions (ex. raw_materials) arrivent avec un index MultiIndex
        # à un seul niveau (chaque clé est un tuple ('Primary Crops - Rice',)) au lieu
        # d'un Index plat de chaînes ('Primary Crops - Rice'). C'est un artefact de
        # cast_index_to_multiindex() côté pipeline d'extraction (sa contrepartie
        # convert_single_level_multi_index_to_regular_index() n'est jamais appelée) ;
        # on l'annule ici pour que l'index exporté soit plat.
        if isinstance(df.index, pd.MultiIndex) and df.index.nlevels == 1:
            df.index = df.index.get_level_values(0)

        if "source" in df.index.names:
            df = df.groupby(level="gas").sum()
            df.index.name = "indicator"

        if "Primary Crops - Rice" in df.index and "RMC" not in df.index:
            total = df.sum().rename("RMC").to_frame().T
            total.index.name = "indicator"
            df = pd.concat([df, total])

        filename = os.path.basename(src_pkl)

        # France : seul F_Y_tot.pkl est exporté, dans dom_<ext> de chaque dossier de
        # scénario France dont l'année de base est celle de ces comptes Monde
        # (2015 : années de base et scénarios 2050 ; 2019 : 2019_FR et ses copies).
        if filename == "F_Y_tot.pkl" and ext_name not in FRANCE_EXCLUDED_EXTENSIONS:
            result = transform(df, FRANCE_CODE)
            for path in FRANCE_DIRS[base_year]:
                _save_result(os.path.join(path, f"dom_{ext_name}"), filename, result, FRANCE_CODE)

        # Monde, Europe et régions hors France : produits à partir de Base_year_2019
        # uniquement (les 3 fichiers, dossier <ext_name> sans préfixe dom_).
        if src_dir == SRC_DIR[1]:
            for mode, dst_base in [("monde", DST_MONDE), ("europe", DST_EUROPE)] + REGION_TARGETS:
                _save_result(os.path.join(dst_base, ext_name), filename, transform(df, mode), mode)


def _clean_generated_outputs():
    """Supprime entièrement le dossier extensions/ de chaque sortie Monde/Europe/
    région hors France (2019_W, 2019_EU27, 2019_<code>) avant de le régénérer.

    Ces dossiers sont entièrement produits par ce script : on peut donc les
    recréer de zéro sans perdre de données externes. Les dossiers de scénario
    France (FRANCE_DIRS, dont 2019_FR) ne sont jamais supprimés : ils contiennent
    déjà des sous-dossiers dom_{ext}/imp_{ext} (d_cba.pkl, F_x_dom.pkl...)
    provenant du moteur France, dans lesquels ce script ne fait qu'ajouter
    dom_{ext}/F_Y_tot.pkl. Les supprimer effacerait ces données externes (vécu :
    un run précédent a écrasé 2019_FR de cette façon).
    """
    for _mode, dst_base in [("monde", DST_MONDE), ("europe", DST_EUROPE)] + REGION_TARGETS:
        if os.path.isdir(dst_base):
            shutil.rmtree(dst_base)


def main():
    _clean_generated_outputs()

    for src_dir in SRC_DIR:
        extensions = [
            d for d in os.listdir(src_dir)
        ]
        print(f"{len(extensions)} extension(s) trouvée(s) : {extensions}\n")

        for ext in extensions:
            print(f"--- {ext} ---")
            process_extension(ext, src_dir)
            print()

    print("Terminé.")


if __name__ == "__main__":
    main()
