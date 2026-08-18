"""
Chemins, constantes et paramètres partagés par toutes les analyses PlaneFR.

Note sur les seuils "absolus" vs "par habitant" :
le fichier seuils.xlsx (feuille "Synthèse") contient trois jeux de lignes pour
les mêmes notions (Lower Safe Bound, Safe Limit, Upper Safe Bound, unité
d'affichage, conversion) : un jeu en valeurs absolues et un jeu par habitant
(suffixe "(p.cap)"). Les figures "multi-scénarios" (valeurs absolues) et "par
habitant" (CBA/PBA) doivent donc passer le bon jeu de constantes aux
fonctions de planefr_lib.
"""

import os
from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
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
# 3 seuils par sous-processus, du plus strict au plus laxiste -- noms de figure
# repris des lignes "brutes" de seuils.xlsx (WORLD_BUDGET_ROW_LOWER/LB/UB
# ci-dessous) : LOWER ("Lower Safe Bound", nouvelle ligne, ligne "Equality
# (Lower bound)") < LB ("Safe Limit", seuil historique, ligne "Equality") < UB
# ("Upper Safe Bound", ligne "Equality (Upper bound)").

# Valeurs absolues (figures multi-scénarios : Synthèse, Overshoot)
THRESHOLD_LOWER_ABS = "Equality (Lower bound)"
THRESHOLD_LB_ABS = "Equality"
THRESHOLD_UB_ABS = "Equality (Upper bound)"
UNIT_ROW_ABS = "Figures unit"
CONVERSION_ROW_ABS = "Unit conversion Exiobase"

# Par habitant (figure comparaison CBA/PBA)
THRESHOLD_LOWER_PER_CAPITA = "Equality (Lower bound) (p. cap)"
THRESHOLD_LB_PER_CAPITA = "Equality (p.cap)"
THRESHOLD_UB_PER_CAPITA = "Equality (Upper bound) (p.cap)"
UNIT_ROW_PER_CAPITA = "Figures unit (p.cap)"
CONVERSION_ROW_PER_CAPITA = "Unit conversion Exiobase (p.cap)"

# threshold_lb_row (_ABS ou _PER_CAPITA) -> ligne LOWER correspondante : permet
# aux fonctions qui reçoivent threshold_lb_row en paramètre générique (ex.
# plot_overshoot.create_overshoot_safe_space_figure) de retrouver le 3e seuil
# sans paramètre dédié.
THRESHOLD_LOWER_FOR = {
    THRESHOLD_LB_ABS: THRESHOLD_LOWER_ABS,
    THRESHOLD_LB_PER_CAPITA: THRESHOLD_LOWER_PER_CAPITA,
}

# ============================================================================
# SEUILS RECALCULÉS SELON UN PRINCIPE DE PARTAGE ("sharing_principle")
# ============================================================================
# Par défaut (sharing_principle=None), les figures lisent LOWER/LB/UB statiquement
# dans seuils.xlsx via les lignes _ABS/_PER_CAPITA ci-dessus (comportement
# historique). Si sharing_principle vaut "Equality 2050"/"Equality 2100"/"Equality
# 2019", LOWER/LB/UB sont recalculés à la volée (partage égal per capita du budget
# mondial) via processing.compute_sharing_seuil, en remplacement de la lecture
# statique de "Equality (Lower bound)"/"Equality"/"Equality (Upper bound)" (et
# leurs variantes par habitant).

# Ligne (ou année, pour "Equality 2019") de seuils.xlsx/feuille "Population" à utiliser
# comme référence de population Monde/France pour chaque principe de partage.
SHARING_PRINCIPLE_POPULATION_ROW = {
    "Equality 2050": "Moyenne 2019-2050",
    "Equality 2100": "Moyenne 2019-2100",
    "Equality 2019": 2019,
}

# Lignes de seuils.xlsx (feuille "Synthèse") donnant le budget mondial LOWER/LB/UB,
# dans l'unité "Unit budget" (échelle mondiale, ex. GtCO2eq) -- différente de
# "Figures unit"/"Figures unit (p.cap)" utilisées partout ailleurs dans le pipeline.
WORLD_BUDGET_ROW_LOWER = "Lower safe bound"
WORLD_BUDGET_ROW_LB = "Safe limit"
WORLD_BUDGET_ROW_UB = "Upper safe bound"

# Facteurs de conversion "Unit budget" (mondiale) -> "Figures unit"/"Figures unit
# (p.cap)", par sous-processus (ex. GHG emissions : GtCO2eq -> MtCO2eq ou tCO2eq) :
# mêmes lignes utilisées en diviseur, comme CONVERSION_ROW_ABS/CONVERSION_ROW_PER_CAPITA.
WORLD_CONVERSION_ROW_ABS = "Unit conversion budget"
WORLD_CONVERSION_ROW_PER_CAPITA = "Unit conversion budget (p. cap)"

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
# COULEUR DU 3E SEUIL ("Lower Safe Bound", THRESHOLD_LOWER_ABS/PER_CAPITA)
# ============================================================================
# Vert clair, distinct du vert du seuil historique (LB, "Safe Limit") utilisé
# par plot_overshoot/plot_radial_synthesis/plot_stacked_bar -- partagé ici pour
# rester cohérent entre les 3 figures.
LOWER_SAFE_BOUND_COLOR = "#90ee90"

# ============================================================================
# FIGURE OVERSHOOT / SAFE OPERATING SPACE
# ============================================================================

# Bornes relatives (en multiples de la limite basse) du dégradé de fond safe/risk.
SAFE_LIMIT_REL = 1.0
RISK_TRANSITION_REL = 2.0
