"""
Chargement des données sources : fichiers Excel (facteurs de caractérisation,
matrices bridge, seuils, population) et pickles MRIO par scénario (d_cba, F_x_dom,
F_Y_tot, M).

Convention générale des loaders de pickles : retournent None si le fichier
n'existe pas pour ce scénario/LP (plutôt que de lever une exception), car
l'absence d'un fichier pour certaines combinaisons scénario x sous-processus
est un cas normal (ex. ghg_combustion absent des scénarios World/Europe).
"""

import re
from pathlib import Path

import pandas as pd

from . import config

# ============================================================================
# FICHIERS EXCEL (communs à tous les scénarios)
# ============================================================================


def load_facteurs_carac():
    """Facteurs de caractérisation : colonnes Sous-processus, Processus du système
    Terre (LP), Extensions exiobase, Facteurs de caractérisation."""
    return pd.read_excel(config.FACTEURS_CARAC_FILE)


def load_bridge_matrices():
    """Matrice bridge 'Final conso. cat.' : correspondance secteur -> catégorie de
    consommation (Food/Housing/Mobility/Final goods/Final services), pour les
    barres empilées et la figure overshoot en valeurs absolues."""
    return pd.read_excel(config.BRIDGE_MATRICES_FILE, sheet_name="Final conso. cat.")


def load_bridge_m_matrix():
    """Bridge 'M' (200 produits x 5 catégories) pour l'agrégation du bubble chart
    par sous-processus. À ne pas confondre avec la feuille 'Alimentation' du même
    classeur, qui est une ventilation fine des produits alimentaires (Riz, Bœuf,
    Porc...) sans rapport avec les 5 catégories de consommation."""
    m_raw = pd.read_excel(config.BRIDGE_MATRICES_FILE, sheet_name="M")
    bridge_m = m_raw.iloc[:200, 3:8].copy()
    bridge_m.index = m_raw.iloc[:200, 2].values
    bridge_m = bridge_m.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (bridge_m > 0).astype(int)


def load_seuils():
    """Feuille 'Synthèse' de seuils.xlsx : seuils/limites planétaires par sous-processus.

    Les noms de ligne y sont normalisés (espaces superflus réduits à un seul,
    début/fin retirés) : la feuille contient plusieurs libellés avec des doubles
    espaces ou espaces de fin (ex. "Unit conversion Exiobase  (p.cap)"), qui
    casseraient sinon les lookups par nom exact (cf. planefr_lib.config)."""
    df = pd.read_excel(config.SEUILS_FILE, sheet_name="Synthèse", index_col=0)
    df.index = df.index.map(lambda x: re.sub(r"\s+", " ", x).strip() if isinstance(x, str) else x)
    return df


def load_dls():
    """Feuille 'DLS' de seuils.xlsx : planchers sociaux (decent living standards)."""
    return pd.read_excel(config.SEUILS_FILE, sheet_name="DLS", index_col=0)


def load_population_df():
    """Feuille 'Population' de seuils.xlsx : population en milliers d'habitants,
    indexée par année, une colonne par géographie (France/Europe/Monde)."""
    return pd.read_excel(config.SEUILS_FILE, sheet_name="Population", index_col=0)


# ============================================================================
# Y_CATEGORY
# ============================================================================


def exclude_y_category(df, excluded=("Exports",)):
    """Retire les colonnes dont Y_category est dans `excluded` (par défaut les
    exports, qui ne font pas partie de la demande finale française).

    Ne s'applique qu'aux données déjà reformatées dans data/3.10.2 (label
    "Exports"). Les scripts de reformatage (reformat_d_cba_monde_europe.py)
    travaillent sur les données brutes amont, où le même type de colonne porte
    le label "Exports: Total (fob)" — ce n'est pas un filtre incohérent, ce
    sont deux étages différents du pipeline avec des libellés différents.
    """
    if "Y_category" not in df.columns.names:
        return df
    mask = ~df.columns.get_level_values("Y_category").isin(excluded)
    return df.loc[:, mask]


# ============================================================================
# D_CBA / F_X_DOM / F_Y_TOT — SCÉNARIOS FRANCE (avec séparation dom/imp)
# ============================================================================


def load_d_cba_k_france(scenario_folder_path, lp_name, origin="imp"):
    """Charge {origin}_{lp}/d_cba_k.pkl (intensités par unité) pour un scénario France."""
    folder_name = f"{'imp' if origin == 'imp' else 'dom'}_{lp_name}"
    file_path = Path(scenario_folder_path) / "extensions" / folder_name / "d_cba_k.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_d_cba_france(scenario_folder_path, lp_name, origin):
    """Charge {origin}_{lp}/d_cba.pkl (totaux, pas intensités) pour un scénario
    France, exports exclus."""
    file_path = Path(scenario_folder_path) / "extensions" / f"{origin}_{lp_name}" / "d_cba.pkl"
    if not file_path.exists():
        return None
    return exclude_y_category(pd.read_pickle(file_path))


def load_f_x_dom_france(scenario_folder_path, lp_name):
    """Charge dom_{lp}/F_x_dom.pkl (empreinte production, dom uniquement) pour un scénario France."""
    file_path = Path(scenario_folder_path) / "extensions" / f"dom_{lp_name}" / "F_x_dom.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_y_tot_france(scenario_folder_path, lp_name):
    """Charge dom_{lp}/F_Y_tot.pkl (émissions directes de la demande finale) pour un
    scénario France, si ce fichier existe pour ce LP (ex. présent pour water/land_use)."""
    file_path = Path(scenario_folder_path) / "extensions" / f"dom_{lp_name}" / "F_Y_tot.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


# ============================================================================
# D_CBA / F_X_DOM / F_Y_TOT — SCÉNARIOS WORLD / EUROPE (pas de séparation dom/imp)
# ============================================================================
# Ces scénarios sont déjà agrégés en un seul fichier par LP (voir
# reformat_d_cba_monde_europe.py) : la colonne Y_category y est déjà réduite à
# "all", donc pas besoin d'exclude_y_category ici.


def load_d_cba_world_europe(scenario_folder_path, lp_name):
    """Charge {lp}/d_cba.pkl pour un scénario World/Europe."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "d_cba.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_x_dom_world_europe(scenario_folder_path, lp_name):
    """Charge {lp}/F_x_dom.pkl pour un scénario World/Europe."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "F_x_dom.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_y_tot_world_europe(scenario_folder_path, lp_name):
    """Charge {lp}/F_Y_tot.pkl pour un scénario World/Europe, si présent."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "F_Y_tot.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


# ============================================================================
# MATRICES D'INTENSITÉ M (bubble chart) ET DEMANDE FINALE Y
# ============================================================================


def load_m_matrix(scenario_folder_path, lp_name, kind):
    """Charge une matrice d'intensité environnementale M pour un LP et un type donné.

    Args:
        kind: "dom" (M_k.pkl de dom_{lp}), "imp" (M_k.pkl de imp_{lp}),
            "row" (M_RoW.pkl de imp_{lp}, part "reste du monde" de l'import).

    Returns:
        pd.DataFrame ou None si le fichier n'existe pas.
    """
    if kind == "dom":
        folder, filename = f"dom_{lp_name}", "M_k.pkl"
    elif kind == "imp":
        folder, filename = f"imp_{lp_name}", "M_k.pkl"
    elif kind == "row":
        folder, filename = f"imp_{lp_name}", "M_RoW.pkl"
    else:
        raise ValueError("kind must be 'dom', 'imp', or 'row'")

    file_path = Path(scenario_folder_path) / "extensions" / folder / filename
    return pd.read_pickle(file_path) if file_path.exists() else None


def get_y_blocks(scenario_folder_path):
    """Charge system/Y.pkl pour un scénario et retourne (y_dom_5, y_imp_5) : deux
    DataFrames 200 x 5 (5 colonnes identiques, dupliquées pour compatibilité avec
    le pipeline de pondération par catégorie) utilisés pour dimensionner/pondérer
    le bubble chart.
    """
    y_path = Path(scenario_folder_path) / "system" / "Y.pkl"
    if not y_path.exists():
        raise FileNotFoundError(f"Y.pkl introuvable : {y_path}")

    y_df = pd.read_pickle(y_path)
    y_sum_0_6 = y_df.iloc[:, 0:6].sum(axis=1)

    y_5cols = pd.concat([y_sum_0_6] * 5, axis=1)
    y_5cols.columns = [f"Y_sum_0_6_{i + 1}" for i in range(5)]

    y_dom_5 = y_5cols.iloc[0:200].copy()
    y_imp_5 = y_5cols.iloc[200:400].copy()
    return y_dom_5, y_imp_5
