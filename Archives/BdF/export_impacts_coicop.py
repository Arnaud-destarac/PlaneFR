"""
Empreinte consumption-based (CBA) d'un scénario PlaneFR, ventilée par poste
COICOP de niveau 1, au format de impacts_IC.csv (une colonne par sous-processus,
comme sur les figures).

Le calcul reprend celui des figures (planefr_lib.processing.process_scenario) :
d_cba_k dom + imp de chaque LP du sous-processus, filtrés et pondérés par les
facteurs de caractérisation, puis divisés par "Unit conversion Exiobase" de
seuils.xlsx (valeurs absolues, en "Figures unit"). Seule différence : au lieu
d'appliquer la bridge par catégorie de consommation (Food, Housing...), on garde
le détail par produit Exiobase, qu'on agrège ensuite par COICOP via
correspondance_exiobase_coicop.csv. Le périmètre de demande finale est celui de
la bridge (ménages + administrations publiques, exports exclus).

F_Y (émissions directes de la demande finale, F_Y_tot.pkl) n'est rattaché à aucun
produit : il est alloué en bloc au poste F_Y_COICOP (Logement).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_DIR.parent / "Module_PlaneFR" / "notebooks"))

from planefr_lib import config, io, processing  # noqa: E402

# ============================================================================
# PARAMÈTRES
# ============================================================================

# Scénario : relatif à DATA_DIR/3.11.2, surchargeable par la variable d'environnement
# PLANEFR_SCENARIO (ex. "SSP2_19_Sus_2050/S1_2050"), utilisée par BdF.ps1
SCENARIO = os.environ.get("PLANEFR_SCENARIO", "SSP2_19_Sus_2050/S1_2050")
SCENARIO_FOLDER = config.DATA_DIR / "3.11.2" / SCENARIO
if not SCENARIO_FOLDER.is_dir():
    raise FileNotFoundError(f"Dossier de scénario introuvable : {SCENARIO_FOLDER}")
CORRESPONDANCE_FILE = CODE_DIR / "ademe_eemrio" / "correspondance_exiobase_coicop.csv"
OUTPUT_PATH = CODE_DIR / "ademe_eemrio" / "impacts.csv"

# Titre de la 1re ligne du CSV (sautée à la lecture, cf. skiprows=1 dans imputation_impacts.ipynb)
TITLE = "Menages et administrations publiques"

# Poste COICOP auquel est alloué F_Y (non ventilable par produit)
F_Y_COICOP = "C04"


# ============================================================================
# CALCUL PAR PRODUIT
# ============================================================================


def load_final_demand_weights(bridge_matrices_df):
    """Poids (0/1) de chacune des 1000 colonnes de d_cba_k (200 produits x 5
    sous-catégories de demande finale) dans les figures : somme de la ligne de
    la bridge, 1 pour ménages/administrations, 0 pour les exports.

    Returns:
        pd.Series indexée par le nom de produit (labels dupliqués, ordre de d_cba_k).
    """
    bridge_categories = bridge_matrices_df.iloc[1:, 6:].apply(pd.to_numeric).fillna(0)
    return pd.Series(bridge_categories.sum(axis=1).values, index=bridge_matrices_df.iloc[1:, 5].values)


def process_subprocess_by_product(subprocess_name, lp_list, scenario_folder_path,
                                  facteurs_carac_df, weights, seuils_df):
    """Empreinte CBA (dom + imp) d'un sous-processus par produit Exiobase, et
    valeur F_Y, dans l'unité des figures. Mêmes LP retenus/écartés que
    processing.process_single_subprocess_scenario.

    Returns:
        tuple: (pd.Series 200 produits, float F_Y), ou (None, 0.0) si aucune donnée.
    """
    extension_factor_map = processing.build_extension_factor_map(facteurs_carac_df, subprocess_name)
    by_product = None
    f_y_value = 0.0

    for lp_name in lp_list:
        d_cba_imp = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="imp")
        d_cba_dom = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="dom")
        if d_cba_imp is None or d_cba_dom is None:
            continue

        vector_imp = processing.filter_and_weight_vector(d_cba_imp, extension_factor_map)
        vector_dom = processing.filter_and_weight_vector(d_cba_dom, extension_factor_map)
        if vector_imp.empty or vector_dom.empty:
            continue

        # Alignement positionnel avec la bridge, comme dans process_scenario
        for vector in (vector_imp, vector_dom):
            if not np.array_equal(vector.index.values, weights.index.values):
                raise ValueError(f"Colonnes de d_cba_k ({lp_name}) non alignées avec la bridge")

        weighted = pd.Series((vector_dom.values + vector_imp.values) * weights.values, index=weights.index)
        lp_by_product = weighted.groupby(level=0, sort=False).sum()
        by_product = lp_by_product if by_product is None else by_product.add(lp_by_product, fill_value=0)

        f_y_tot_df = io.load_f_y_tot_france(scenario_folder_path, lp_name)
        if f_y_tot_df is not None:
            f_y_value += float(processing.filter_and_weight_vector(f_y_tot_df, extension_factor_map).sum())

    if by_product is None:
        return None, 0.0

    conversion_factor = processing.lookup_seuil(seuils_df, config.CONVERSION_ROW_ABS, subprocess_name)
    if conversion_factor is not None:
        by_product = by_product / conversion_factor
        f_y_value = f_y_value / conversion_factor
    return by_product, f_y_value


# ============================================================================
# AGRÉGATION COICOP ET EXPORT
# ============================================================================


def load_correspondance():
    """Correspondance produit Exiobase -> (coicop_1, libellé). Le libellé d'un
    code est le plus fréquent dans le fichier (quelques lignes y portent le
    libellé d'un autre poste)."""
    corr = pd.read_csv(CORRESPONDANCE_FILE, sep=";")
    labels = corr.groupby("coicop_1")["Postes_coicop_1"].agg(lambda s: s.mode().iloc[0])
    return corr.set_index("ExioName")["coicop_1"], labels


def column_name(subprocess_name, seuils_df):
    unit = seuils_df.loc[config.UNIT_ROW_ABS, subprocess_name] if subprocess_name in seuils_df.columns else None
    return f"{subprocess_name} ({unit})" if isinstance(unit, str) and unit.strip() else subprocess_name


def format_number(value):
    return f"{value:.12g}".replace(".", ",")


def write_impacts_csv(table, labels, output_path):
    """Écrit `table` (index coicop_1, une colonne par sous-processus) au format
    de impacts_IC.csv : ligne de titre, en-tête, postes, ligne vide, Total."""
    n_cols = 2 + len(table.columns)
    empty_row = ";" * (n_cols - 1)
    lines = [TITLE + empty_row, ";".join(["coicop_1", "Postes_coicop_1", *table.columns])]
    for code, row in table.iterrows():
        lines.append(";".join([code, labels[code], *map(format_number, row.values)]))
    lines.append(empty_row)
    lines.append(";".join(["Total", "", *map(format_number, table.sum().values)]))
    with open(output_path, "w", encoding="utf-8", newline="\r\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    facteurs_carac_df = io.load_facteurs_carac()
    bridge_matrices_df = io.load_bridge_matrices()
    seuils_df = io.load_seuils()
    product_to_coicop, labels = load_correspondance()
    weights = load_final_demand_weights(bridge_matrices_df)

    missing = set(weights.index) - set(product_to_coicop.index)
    if missing:
        raise ValueError(f"Produits absents de {CORRESPONDANCE_FILE.name} : {sorted(missing)}")

    columns = {}
    for subprocess_name, lp_list in processing.get_unique_subprocesses(facteurs_carac_df).items():
        by_product, f_y_value = process_subprocess_by_product(
            subprocess_name, lp_list, SCENARIO_FOLDER, facteurs_carac_df, weights, seuils_df,
        )
        if by_product is None:
            print(f"  {subprocess_name} : aucune donnée, ignoré")
            continue
        by_coicop = by_product.groupby(product_to_coicop).sum()
        by_coicop[F_Y_COICOP] += f_y_value
        columns[column_name(subprocess_name, seuils_df)] = by_coicop

    table = pd.DataFrame(columns).fillna(0.0).sort_index()

    # Contrôle : totaux identiques à ceux des figures (process_scenario)
    figures_data, _ = processing.process_scenario(SCENARIO_FOLDER, facteurs_carac_df, bridge_matrices_df, seuils_df)
    for subprocess_name, payload in figures_data.items():
        figure_total = float(payload["domestique"].sum() + payload["importé"].sum())
        total = table[column_name(subprocess_name, seuils_df)].sum()
        status = "OK" if np.isclose(total, figure_total, rtol=1e-9) else "ÉCART"
        print(f"  {status:5s} {subprocess_name:28s} {total:14.4f}  (figures : {figure_total:.4f})")

    write_impacts_csv(table, labels, OUTPUT_PATH)
    print(f"Sauvegardé : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
