"""
Chemins, constantes et paramètres partagés par toutes les analyses PlaneFR.

Note sur les seuils "absolus" vs "par habitant" :
le fichier seuils.xlsx contient deux jeux de lignes pour les mêmes notions
(limite basse, limite haute, unité d'affichage, conversion) : un jeu en
valeurs absolues et un jeu par habitant (suffixe "(p. hab)"). Les figures
"multi-scénarios" (valeurs absolues) et "par habitant" (CBA/PBA) doivent
donc passer le bon jeu de constantes aux fonctions de planefr_lib.
"""

import os
from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================

PROJECT_DIR = Path(r"C:\Users\Arnaud\Documents\PlaneFR")
DATA_DIR = PROJECT_DIR / "data"
BASE_DATA_DIR = DATA_DIR / "3.10.2"

# Redirigeable via la variable d'environnement PLANEFR_FIGURES_DIR (utile pour
# vérifier qu'un notebook s'exécute correctement sans écraser les figures
# actuelles) ; par défaut, le dossier figures/ du projet comme d'habitude.
FIGURES_DIR = Path(os.environ.get("PLANEFR_FIGURES_DIR", str(PROJECT_DIR / "figures")))

FACTEURS_CARAC_FILE = DATA_DIR / "facteurs de caractérisation.xlsx"
BRIDGE_MATRICES_FILE = DATA_DIR / "bridge_matrices.xlsx"
SEUILS_FILE = DATA_DIR / "seuils.xlsx"

# ============================================================================
# SCÉNARIOS
# ============================================================================

# Index du scénario de référence dans get_scenario_folders() : "Base_year_2015"
# est toujours le premier par ordre alphabétique.
REFERENCE_SCENARIO_IDX = 0


def get_scenario_folders(exclude_2019=False, exclude_others=False):
    """Liste les dossiers de scénarios présents dans BASE_DATA_DIR, triés par nom.

    Args:
        exclude_2019: si True, exclut les dossiers "2019_*" (France/World/Europe_27
            en géographies alternatives) pour ne garder que base-year + scénarios 2050
            — c'est le comportement voulu pour les figures multi-scénarios en valeurs
            absolues. Si False, les inclut — nécessaire pour la figure par habitant
            CBA/PBA, qui compare aussi France vs Europe vs Monde.

    Returns:
        list[Path]: dossiers de scénarios triés par nom.
    """
    folders = [d for d in BASE_DATA_DIR.iterdir() if d.is_dir()]
    if exclude_2019:
        folders = [d for d in folders if "2019" not in d.name]
    if exclude_others:
        folders = [d for d in folders if "2019" in d.name]
    return sorted(folders)


# ============================================================================
# SEUILS : noms des lignes de seuils.xlsx (feuille "Synthèse")
# ============================================================================

# Valeurs absolues (figures multi-scénarios : Synthèse, Overshoot)
THRESHOLD_LB_ABS = "Égalité"
THRESHOLD_UB_ABS = "Égalité (limite haute)"
UNIT_ROW_ABS = "Unité affichage"
CONVERSION_ROW_ABS = "Conversion"

# Par habitant (figure comparaison CBA/PBA)
THRESHOLD_LB_PER_CAPITA = "Égalité (p. hab)"
THRESHOLD_UB_PER_CAPITA = "Égalité (p. hab) limite haute"
UNIT_ROW_PER_CAPITA = "Unité affichage (p.hab)"
CONVERSION_ROW_PER_CAPITA = "Conversion (p.hab)"

# ============================================================================
# COULEURS DES CATÉGORIES DE CONSOMMATION
# ============================================================================

CATEGORY_COLORS = {
    0: "#1f77b4",  # Bleu
    1: "#ff7f0e",  # Orange
    2: "#2ca02c",  # Vert
    3: "#d62728",  # Rouge
    4: "#9467bd",  # Violet
}

# Ordre canonique d'affichage des catégories (F_Y toujours en dernier, bien identifiable)
CATEGORY_ORDER = ["Food", "Housing", "Mobility", "Final goods", "Final services", "F_Y"]

# Couleur stable par nom de catégorie (indépendante de la position après agrégation multi-LP)
CATEGORY_COLOR_MAP = {
    "Food": "#1f77b4",
    "Housing": "#ff7f0e",
    "Mobility": "#2ca02c",
    "Final goods": "#d62728",
    "Final services": "#9467bd",
    "F_Y": "#7f7f7f",
}

# ============================================================================
# FIGURE OVERSHOOT / SAFE OPERATING SPACE
# ============================================================================

# Bornes relatives (en multiples de la limite basse) du dégradé de fond safe/risk.
SAFE_LIMIT_REL = 1.0
RISK_TRANSITION_REL = 2.0
