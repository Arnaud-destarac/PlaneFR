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
CONVERSION_ROW_ABS = "Conversion exiobase"

# Par habitant (figure comparaison CBA/PBA)
THRESHOLD_LB_PER_CAPITA = "Égalité (p. hab)"
THRESHOLD_UB_PER_CAPITA = "Égalité (p. hab) limite haute"
UNIT_ROW_PER_CAPITA = "Unité affichage (p.hab)"
CONVERSION_ROW_PER_CAPITA = "Conversion exiobase (p.hab)"

# ============================================================================
# SEUILS RECALCULÉS SELON UN PRINCIPE DE PARTAGE ("sharing_principle")
# ============================================================================
# Par défaut (sharing_principle=None), les figures lisent LB/UB statiquement dans
# seuils.xlsx via les lignes _ABS/_PER_CAPITA ci-dessus (comportement historique).
# Si sharing_principle vaut "Equality 2050"/"Equality 2100"/"Equality 2019", LB/UB sont
# recalculés à la volée (partage égal per capita du budget mondial) via
# processing.compute_sharing_seuil, en remplacement de "Égalité"/"Égalité (p. hab)" (et
# leurs variantes limite haute).

# Ligne (ou année, pour "Equality 2019") de seuils.xlsx/feuille "Population" à utiliser
# comme référence de population Monde/France pour chaque principe de partage.
SHARING_PRINCIPLE_POPULATION_ROW = {
    "Equality 2050": "Moyenne 2019-2050",
    "Equality 2100": "Moyenne 2019-2100",
    "Equality 2019": 2019,
}

# Lignes de seuils.xlsx (feuille "Synthèse") donnant le budget mondial LB/UB, dans
# l'unité "Unité" (échelle mondiale, ex. GtCO2eq) -- différente de "Unité affichage"/
# "Unité affichage (p.hab)" utilisées partout ailleurs dans le pipeline.
WORLD_BUDGET_ROW_LB = "Budget Global annuel"
WORLD_BUDGET_ROW_UB = "Limite haute"

# Facteurs de conversion "Unité" (mondiale) -> "Unité affichage"/"Unité affichage (p.hab)",
# par sous-processus (ex. GHG emissions : GtCO2eq -> MtCO2eq ou tCO2eq) : mêmes lignes
# utilisées en diviseur, comme CONVERSION_ROW_ABS/CONVERSION_ROW_PER_CAPITA.
WORLD_CONVERSION_ROW_ABS = "Conversion budget"
WORLD_CONVERSION_ROW_PER_CAPITA = "Conversion budget (p.hab)"

# ============================================================================
# COULEURS DES CATÉGORIES DE CONSOMMATION
# ============================================================================

CATEGORY_COLORS = {
    0: "#6ac1ff",  # Bleu
    1: "#fba65b",  # Orange
    2: "#4fff4f",  # Vert
    3: "#ff6d6d",  # Rouge
    4: "#d0a3fa",  # Violet
}

# Ordre canonique d'affichage des catégories (F_Y toujours en dernier, bien identifiable)
CATEGORY_ORDER = ["Food", "Housing", "Mobility", "Final goods", "Final services", "F_Y"]

# Couleur stable par nom de catégorie (indépendante de la position après agrégation multi-LP)
CATEGORY_COLOR_MAP = {
    0: "#84c9fa",  # Bleu
    1: "#f9c18f",  # Orange
    2: "#8cfc8c",  # Vert
    3: "#fd9393",  # Rouge
    4: "#d0a3fa",  # Violet
    "F_Y": "#7f7f7f",
}

# ============================================================================
# FIGURE OVERSHOOT / SAFE OPERATING SPACE
# ============================================================================

# Bornes relatives (en multiples de la limite basse) du dégradé de fond safe/risk.
SAFE_LIMIT_REL = 1.0
RISK_TRANSITION_REL = 2.0
