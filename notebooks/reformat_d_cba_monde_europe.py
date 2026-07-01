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
SRC_DIR  = r"C:\Users\Arnaud\Documents\PlaneFR\data\Monde\base-year_2019\extensions"
DST_MONDE   = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_World\extensions"
DST_EUROPE  = r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2\2019_Europe_27\extensions"

# Y_categories à exclure
YCATS_EXCLUDED = {"Exports: Total (fob)"}

# Nombre de régions européennes (les 27 premières dans l'ordre du MultiIndex)
N_EU_REGIONS = 27


def transform(df: pd.DataFrame, mode: str, src_pkl: str) -> pd.DataFrame:
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
        eu_regions = df.columns.get_level_values("region").unique()[:N_EU_REGIONS]
        mask_region = df.columns.get_level_values("region").isin(eu_regions)
        df = df.loc[:, mask_region]

    # 3. Sommer toutes les colonnes → une valeur par ligne (Series)
    totals = df.sum(axis=1)

    # 4. Reconstruire un DataFrame avec le bon MultiIndex de colonnes
    region_label  = "World" if mode == "monde" else "Europe"
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


def process_extension(ext_name: str):
    srcs_pkl = [os.path.join(SRC_DIR, ext_name, "d_cba.pkl"), os.path.join(SRC_DIR, ext_name, "F_x_dom.pkl")]

    for src_pkl in srcs_pkl:
        if not os.path.exists(src_pkl):
            print(f"  [SKIP] {os.path.basename(src_pkl)} introuvable dans {ext_name}")
            return

        with open(src_pkl, "rb") as f:
            df = pickle.load(f)

        if "source" in df.index.names:
            df = df.groupby(level="gas").sum()
            df.index.name = "indicator"
        
        if "Primary Crops - Rice" in df.index:
            df = df.sum().rename("RMC").to_frame().T
            df.index.name = "indicator"

        for mode, dst_base in [("monde", DST_MONDE), ("europe", DST_EUROPE)]:
            out_dir = os.path.join(dst_base, ext_name)
            os.makedirs(out_dir, exist_ok=True)
            out_pkl = os.path.join(out_dir, src_pkl.split(os.sep)[-1])  # même nom de fichier que src_pkl

            result = transform(df, mode, src_pkl)
            with open(out_pkl, "wb") as f:
                pickle.dump(result, f)

            print(f"  [{mode.upper()}] {out_pkl}  shape={result.shape}")


def main():
    extensions = [
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d))
    ]
    print(f"{len(extensions)} extension(s) trouvée(s) : {extensions}\n")

    for ext in extensions:
        print(f"--- {ext} ---")
        process_extension(ext)
        print()

    print("Terminé.")


if __name__ == "__main__":
    main()
