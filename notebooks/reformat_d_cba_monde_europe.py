"""
Pour chaque extension dans data/Monde/base-year_2015 et base-year_2019/extensions :
- transforme d_cba.pkl / F_x_dom.pkl / F_Y_tot.pkl en versions Monde, Europe et France
- Monde  : supprime les Y_category indésirables, somme tout -> (region="World", Y_category="all", sector="all")
- Europe : idem mais en ne gardant que les 27 premières régions -> (region="Europe", ...)
- France : idem mais en ne gardant que la région "FR" -> (region="France", ...)

base-year_2015 (SRC_DIR[0]) ne produit qu'une version France, copiée dans tous
les autres dossiers de scénarios déjà présents dans data/3.10.2 (dom_{ext}) ;
base-year_2019 (SRC_DIR[1]) produit les trois versions dans leurs dossiers dédiés.
"""

import os
import pickle
import pandas as pd

# Chemins
SRC_DIR = [
    r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2015\extensions",
    r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2019\extensions",
]
DST_MONDE = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_World\extensions"
DST_EUROPE = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_Europe_27\extensions"
DST_FRANCE = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_France\extensions"

# Découvrir dynamiquement les autres dossiers de scénarios déjà présents dans 3.10.2
DATA_DIR = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2"
EXCLUDED_DIRS = {"2019_World", "2019_Europe_27", "2019_France"}
OTHER_DIRS = [
    os.path.join(DATA_DIR, d, "extensions")
    for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d, "extensions")) and d not in EXCLUDED_DIRS
]

# Y_categories à exclure. Ce label ("Exports: Total (fob)") est celui des données
# BRUTES en amont (SRC_DIR) ; il ne faut pas le confondre avec le label "Exports"
# utilisé par les notebooks d'analyse (planefr_lib.io.exclude_y_category), qui
# lisent eux les données déjà reformatées dans data/3.10.2 — deux étages du
# pipeline, deux libellés différents pour le même concept, pas une incohérence.
YCATS_EXCLUDED = {"Exports: Total (fob)"}

# Liste des 27 régions européennes (UE) issue du fichier de référence
DETAIL_LEVELS_PATH = r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2019\system\detail_levels.xlsx"
with open(DETAIL_LEVELS_PATH, "rb") as f:
    _regions_df = pd.read_excel(f, sheet_name="regions")
EU_REGIONS = _regions_df["region"].astype(str).to_list()[:27]


def transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode='monde'  -> toutes les régions sauf Y_cats exclus, somme -> World/all/all
    mode='europe' -> 27 premières régions sauf Y_cats exclus, somme -> Europe/all/all
    mode='france' -> région FR uniquement, somme -> France/all/all
    """
    if "Y_category" in df.columns.names:
        mask_ycat = ~df.columns.get_level_values("Y_category").isin(YCATS_EXCLUDED)
        df = df.loc[:, mask_ycat]

    if mode == "europe":
        df = df.loc[:, df.columns.get_level_values("region").isin(EU_REGIONS)]
    elif mode == "france":
        df = df.loc[:, df.columns.get_level_values("region") == "FR"]

    totals = df.sum(axis=1)

    region_label = {"monde": "World", "europe": "Europe", "france": "France"}[mode]
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

        if "source" in df.index.names:
            df = df.groupby(level="gas").sum()
            df.index.name = "indicator"

        if "Primary Crops - Rice" in df.index:
            df = df.sum().rename("RMC").to_frame().T
            df.index.name = "indicator"

        filename = os.path.basename(src_pkl)

        if src_dir == SRC_DIR[0]:
            # base-year_2015 : une seule version France, dupliquée dans chaque
            # dossier de scénario déjà présent (ils partagent tous le même base year).
            result = transform(df, "france")
            for path in OTHER_DIRS:
                _save_result(os.path.join(path, f"dom_{ext_name}"), filename, result, "france")
        else:
            for mode, dst_base in [("monde", DST_MONDE), ("europe", DST_EUROPE), ("france", DST_FRANCE)]:
                if mode == "france" and (filename != "F_Y_tot.pkl" or ext_name == "ghg_emissions"):
                    continue
                out_dir = os.path.join(dst_base, ext_name if mode in ("europe", "monde") else f"dom_{ext_name}")
                _save_result(out_dir, filename, transform(df, mode), mode)


def main():
    for src_dir in SRC_DIR:
        extensions = [
            d for d in os.listdir(src_dir)
            if os.path.isdir(os.path.join(src_dir, d)) and d != "air_emissions"
        ]
        print(f"{len(extensions)} extension(s) trouvée(s) : {extensions}\n")

        for ext in extensions:
            print(f"--- {ext} ---")
            process_extension(ext, src_dir)
            print()

    print("Terminé.")


if __name__ == "__main__":
    main()
