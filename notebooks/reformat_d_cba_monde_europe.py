"""
Pour chaque extension dans data/Monde/base-year_2019/extensions :
- transforme d_cba.pkl en version Monde et Europe
- Monde  : supprime les Y_category indésirables, somme tout → (region="World", Y_category="all", sector="all")
- Europe : idem mais en ne gardant que les 27 premières régions → (region="Europe", ...)
"""

import os
import pickle
import pandas as pd

# Chemins
SRC_DIR  = [r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2015\extensions",
            r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2019\extensions"]
DST_MONDE   = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_World\extensions"
DST_EUROPE  = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_Europe_27\extensions"
DST_FRANCE =    r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_France\extensions"

# Découvrir dynamiquement les autres dossiers dans 3.10.2
DATA_DIR = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2"
EXCLUDED_DIRS = {"2019_World", "2019_Europe_27", "2019_France"}
OTHER_DIRS = [
    os.path.join(DATA_DIR, d, "extensions")
    for d in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, d, "extensions")) and d not in EXCLUDED_DIRS
]

# Y_categories à exclure
YCATS_EXCLUDED = {"Exports: Total (fob)"}

# Liste des 27 régions européennes (UE) issue du fichier de référence
DETAIL_LEVELS_PATH = r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\Monde\base-year_2019\system\detail_levels.xlsx"
with open(DETAIL_LEVELS_PATH, "rb") as f:
    _regions_df = pd.read_excel(f, sheet_name="regions")
EU_REGIONS = _regions_df["region"].astype(str).to_list()[:27]


def transform(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode='monde'  → toutes les régions sauf Y_cats exclus, somme → World/all/all
    mode='europe' → 27 premières régions sauf Y_cats exclus, somme → Europe/all/all
    """
    # 1. Filtrer les Y_categories indésirables
    if "Y_category" in df.columns.names:
        mask_ycat = ~df.columns.get_level_values("Y_category").isin(YCATS_EXCLUDED)
        df = df.loc[:, mask_ycat]

    # 2. Pour Europe : garder seulement les 27 premières régions
    if mode == "europe":
        mask_region = df.columns.get_level_values("region").isin(EU_REGIONS)
        df = df.loc[:, mask_region]

    if mode == "france":
        mask_region = df.columns.get_level_values("region") ==  "FR"
        df = df.loc[:, mask_region]

    # 3. Sommer toutes les colonnes → une valeur par ligne (Series)
    totals = df.sum(axis=1)

    # 4. Reconstruire un DataFrame avec le bon MultiIndex de colonnes
    region_label  = "World" if mode == "monde" else "Europe"
    if mode == "france":
        region_label = "France"
    if "Y_category" in df.columns.names:
        col_index = pd.MultiIndex.from_tuples(
            [(region_label, "all", "all")],
            names=["region", "Y_category", "sector"],
        )
    else : 
        col_index = pd.MultiIndex.from_tuples(
            [(region_label, "all")],
            names=["region", "sector"],
        )
    result = pd.DataFrame(totals.values, index=df.index, columns=col_index)
    return result


def process_extension(ext_name: str, src_dir: str):
    srcs_pkl = [os.path.join(src_dir, ext_name, "d_cba.pkl"), os.path.join(src_dir, ext_name, "F_x_dom.pkl"), os.path.join(src_dir, ext_name, "F_Y_tot.pkl")]

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

        if src_dir == SRC_DIR[0] :
            result = transform(df, "france")
            for path in OTHER_DIRS:
                out_dir = os.path.join(path, f"dom_{ext_name}")
                os.makedirs(out_dir, exist_ok=True)
                out_pkl = os.path.join(out_dir, src_pkl.split(os.sep)[-1])  # même nom de fichier que src_pkl

                with open(out_pkl, "wb") as f:
                    pickle.dump(result, f)
                
                print(f"  [{"france".upper()}] {out_pkl}  shape={result.shape}")
        else: 
            for mode, dst_base in [("monde", DST_MONDE), ("europe", DST_EUROPE), ("france", DST_FRANCE)]:
                if mode == "france" and (src_pkl.split(os.sep)[-1] != "F_Y_tot.pkl" or ext_name == "ghg_emissions"):
                    continue

                if mode == "europe" or mode == "monde":
                    out_dir = os.path.join(dst_base, ext_name)
                else:
                    out_dir = os.path.join(dst_base, f"dom_{ext_name}")
                os.makedirs(out_dir, exist_ok=True)
                out_pkl = os.path.join(out_dir, src_pkl.split(os.sep)[-1])  # même nom de fichier que src_pkl

                result = transform(df, mode)
                with open(out_pkl, "wb") as f:
                    pickle.dump(result, f)

                print(f"  [{mode.upper()}] {out_pkl}  shape={result.shape}")


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
