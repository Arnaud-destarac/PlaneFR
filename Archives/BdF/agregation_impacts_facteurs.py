"""
Agrégation des impacts par ménage (impacts_facteurs.csv, sortie de
imputation_impacts.ipynb) par catégorie socio-économique.

Les impacts par ménage y sont déjà pondérés par pondmen (leur somme sur tous les
ménages redonne le total national de impacts.csv) : l'impact d'une catégorie est
donc la simple somme des total_{slug} de ses ménages.

Sortie : un classeur Excel (un CSV ne peut pas contenir plusieurs feuilles) avec
une feuille par variable de regroupement, une ligne par catégorie et une colonne
par sous-processus, nommée comme dans impacts.csv (via impacts_facteurs_legende.csv).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

WORK_DIR = Path(__file__).resolve().parent
INPUT_PATH = WORK_DIR / "impacts_facteurs.csv"
LEGENDE_PATH = WORK_DIR / "impacts_facteurs_legende.csv"
OUTPUT_PATH = WORK_DIR / "impacts_facteurs_agrégé.xlsx"

# Feuille -> colonne de regroupement de impacts_facteurs.csv
GROUPINGS = {
    "décile": "decile_uc_2",
    "unité urbaine": "tuu_2",
    "croisé": "decile_tuu",
}

# Poids démographique relatif du ménage (part de la population, somme = 1)
WEIGHT_COL = "weight"

# Feuilles "décile" et "croisé" : regroupement des déciles en tranches
DECILE_BINS = {**{f"d{i}": "13" for i in range(1, 4)},
               **{f"d{i}": "47" for i in range(4, 8)},
               **{f"d{i}": "89" for i in range(8, 10)},
               **{f"d{i}": "10" for i in range(10, 11)}}
# Ordre d'affichage des tranches (le tri naturel mettrait '10' en premier)
DECILE_ORDER = ["13", "47", "89", "10"]

# Feuille "croisé" : regroupement et abréviation des unités urbaines
# ('autre' fusionné avec 'urbain')
TUU_BINS = {"autre": "U", "urbain": "U", "rural": "R"}


def bin_decile(label):
    """'d6' -> '47'."""
    return DECILE_BINS[str(label)]


def bin_decile_tuu(label):
    """'d6_autre' -> '47U', 'd2_rural' -> '13R'."""
    decile, _, tuu = str(label).partition("_")
    return f"{DECILE_BINS[decile]}{TUU_BINS.get(tuu.lower(), tuu)}"


# Feuille -> transformation des catégories avant agrégation
RELABEL = {
    "décile": bin_decile,
    "croisé": bin_decile_tuu,
}


def natural_key(label):
    """Clé de tri qui place d2 avant d10 (d1, d2, ..., d10), et les tranches
    de déciles ('13', '47U', ...) dans l'ordre de DECILE_ORDER."""
    label = str(label)
    if label[:2] in DECILE_ORDER:
        return [DECILE_ORDER.index(label[:2]), label[2:]]
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(label))]


def aggregate(df, group_col, indicator_cols):
    """Part de la population (somme de weight) et somme des impacts par catégorie
    de `group_col`, plus une ligne Total."""
    table = df.groupby(group_col)[[WEIGHT_COL, *indicator_cols]].sum().rename(columns=indicator_cols)
    table = table.loc[sorted(table.index, key=natural_key)]
    table.loc["Total"] = table.sum()
    table.index.name = group_col
    return table


def main():
    df = pd.read_csv(INPUT_PATH, sep=";", decimal=",", encoding="utf-8")
    legende = pd.read_csv(LEGENDE_PATH, sep=";", decimal=",", encoding="utf-8")

    # total_{slug} -> nom du sous-processus tel que dans impacts.csv
    indicator_cols = {f"total_{slug}": name for name, slug in zip(legende["indicateur"], legende["slug"])}
    missing = [col for col in [WEIGHT_COL, *indicator_cols, *GROUPINGS.values()] if col not in df.columns]
    if missing:
        raise ValueError(f"Colonnes absentes de {INPUT_PATH.name} : {missing}")

    national_total = df[list(indicator_cols)].sum().rename(indicator_cols)

    with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
        for sheet_name, group_col in GROUPINGS.items():
            data = df
            if sheet_name in RELABEL:
                data = df.assign(**{group_col: df[group_col].map(RELABEL[sheet_name], na_action="ignore")})
            table = aggregate(data, group_col, indicator_cols)
            # Contrôle : ménages sans catégorie (NaN) exclus du groupby
            if not np.allclose(table.loc["Total", national_total.index], national_total, rtol=1e-9):
                print(f"  ⚠ {sheet_name} : total différent du total national (catégories manquantes ?)")
            table.to_excel(writer, sheet_name=sheet_name)
            print(f"  {sheet_name} ({group_col}) : {len(table) - 1} catégories")

    print(f"Sauvegardé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
