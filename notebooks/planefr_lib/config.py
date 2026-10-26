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
import warnings
from pathlib import Path

# ============================================================================
# CHEMINS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
# Version Exiobase des données : redirigeable via la variable d'environnement
# PLANEFR_EXIOBASE_VERSION (positionnée par les scripts de pipeline de
# matmat-ademe) ; par défaut 3.11.2 comme d'habitude.
BASE_DATA_DIR = DATA_DIR / os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")

# Redirigeable via la variable d'environnement PLANEFR_FIGURES_DIR (utile pour
# vérifier qu'un notebook s'exécute correctement sans écraser les figures
# actuelles) ; par défaut, le dossier figures/ du projet comme d'habitude.
FIGURES_DIR = Path(os.environ.get("PLANEFR_FIGURES_DIR", str(PROJECT_DIR / "figures")))

FACTEURS_CARAC_FILE = DATA_DIR / "facteurs de caractérisation.xlsx"
BRIDGE_MATRICES_FILE = DATA_DIR / "bridge_matrices.xlsx"
SEUILS_FILE = DATA_DIR / "seuils.xlsx"
BUDGET_SHARES_FILE = DATA_DIR / "budget_shares.xlsx"
# Pressions (empreinte CBA) ventilées par catégorie de ménages (enquête Budget
# de famille) : 1 feuille par ventilation ("décile", "unité urbaine", "croisé").
BDF_FILE = PROJECT_DIR.parent / "Github" / "ademe_eemrio" / "impacts_facteurs_agrégé.xlsx"
BDF_WEIGHT_COL = "weight"
BDF_TOTAL_ROW = "Total"

# ============================================================================
# SCÉNARIOS
# ============================================================================

# Index du scénario de référence dans get_scenario_folders() : "Base_year_2015"
# est toujours le premier par ordre alphabétique.
REFERENCE_SCENARIO_IDX = 0

# Nom du dossier du scénario de référence (baseline), attendu dans chaque groupe
# de scénarios (cf. get_scenario_folders).
REFERENCE_SCENARIO_NAME = "Base_year_2015"

# Groupe de scénarios affiché par défaut : sous-dossier de BASE_DATA_DIR
# contenant les scénarios de transition 2050 calculés sur un jeu d'accounts
# donné ("no_SSP" pour les accounts Base_year_2015, "SSP1_2050", "SSP2_2050"...
# pour les accounts projetés selon un SSP, cf. run_engine_batch.ps1 de
# matmat-ademe). Redirigeable via la variable d'environnement
# PLANEFR_SCENARIO_GROUP, ou par notebook via le paramètre scenario_group de
# get_scenario_folders. "" = anciens scénarios placés directement dans
# BASE_DATA_DIR.
SCENARIO_GROUP = os.environ.get("PLANEFR_SCENARIO_GROUP", "no_SSP")


def _is_scenario_folder(path):
    """Un dossier de scénario contient un sous-dossier "extensions" (sortie engine ;
    "system" est absent des dossiers 2019_* hors France), contrairement aux
    dossiers de groupe (no_SSP, SSP1_2050...)."""
    return path.is_dir() and (path / "extensions").is_dir()


def get_scenario_groups():
    """Liste les groupes de scénarios disponibles (sous-dossiers de BASE_DATA_DIR
    qui ne sont pas eux-mêmes des dossiers de scénario), triés par nom."""
    return sorted(
        d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir() and not _is_scenario_folder(d)
    )


def get_scenario_folders(exclude_2019=False, exclude_others=False, scenario_group=None):
    """Liste les dossiers de scénarios du groupe demandé, triés par nom.

    Les scénarios affichés sont tous ceux du dossier BASE_DATA_DIR / scenario_group
    (y compris le scénario de référence, REFERENCE_SCENARIO_NAME, qui doit donc s'y
    trouver). Les dossiers "2019_*" sont toujours lus dans BASE_DATA_DIR.

    Args:
        exclude_2019: si True, exclut les dossiers "2019_*" (France/World/Europe_27
            en géographies alternatives) pour ne garder que base-year + scénarios 2050
            — c'est le comportement voulu pour les figures multi-scénarios en valeurs
            absolues. Si False, les inclut — nécessaire pour la figure par habitant
            CBA/PBA, qui compare aussi France vs Europe vs Monde.
        exclude_others: si True, ne garde que les dossiers "2019_*".
        scenario_group: groupe de scénarios à afficher ("no_SSP", "SSP1_2050"...,
            cf. get_scenario_groups()). None = SCENARIO_GROUP ; "" = scénarios
            placés directement dans BASE_DATA_DIR.

    Returns:
        list[Path]: dossiers de scénarios triés par nom.
    """
    if scenario_group is None:
        scenario_group = SCENARIO_GROUP

    root_folders = [d for d in BASE_DATA_DIR.iterdir() if _is_scenario_folder(d)]
    folders_2019 = [d for d in root_folders if "2019" in d.name]

    if scenario_group:
        group_dir = BASE_DATA_DIR / scenario_group
        if not group_dir.is_dir():
            raise ValueError(
                f"Groupe de scénarios '{scenario_group}' introuvable dans {BASE_DATA_DIR}. "
                f"Groupes disponibles : {get_scenario_groups()}"
            )
        # Tous les scénarios du dossier de groupe, et uniquement eux
        scenario_folders = [d for d in group_dir.iterdir() if _is_scenario_folder(d)]
        if REFERENCE_SCENARIO_NAME not in {d.name for d in scenario_folders}:
            warnings.warn(
                f"Le groupe '{scenario_group}' ne contient pas {REFERENCE_SCENARIO_NAME} : "
                f"le scénario de référence (REFERENCE_SCENARIO_IDX) sera le premier par "
                f"ordre alphabétique."
            )
    else:
        scenario_folders = [d for d in root_folders if "2019" not in d.name]

    if exclude_others:
        folders = folders_2019
    elif exclude_2019:
        folders = scenario_folders
    else:
        folders = scenario_folders + folders_2019
    return sorted(folders, key=lambda d: d.name)


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
# PARTS DU BUDGET MONDIAL PAR RÉGION EXIOBASE (budget_shares.xlsx)
# ============================================================================
# Alternative au partage égal per capita ci-dessus (SHARING_PRINCIPLE_POPULATION_ROW) :
# le budget d'un pays/région est le budget mondial multiplié par une part lue dans
# budget_shares.xlsx, et non plus déduit d'un ratio de populations. Utilisé par
# plot_overshoot.create_overshoot_safe_space_figure_by_region (figure en valeurs
# absolues, donc conversions _ABS : WORLD_CONVERSION_ROW_ABS / CONVERSION_ROW_ABS).

BUDGET_SHARES_SHEET = "Parts_regions_EXIOBASE"

# La feuille commence par un titre et deux lignes de notes : la vraie ligne
# d'en-têtes ("Code EXIOBASE", "Région", "Part EPC ref. ...") est la 4e.
BUDGET_SHARES_HEADER_ROW = 3

# Colonne servant d'index (code région EXIOBASE : FR, EU27, W, CN...).
BUDGET_SHARE_CODE_COLUMN = "Code EXIOBASE"

# Principe de partage -> motifs (regex, insensibles à la casse) identifiant la
# colonne de part à utiliser pour la référence (position des bulles) et pour les
# deux bornes de sensibilité.
# Les motifs "ref" sont ancrés sur "Part ..." à dessein : la feuille contient
# aussi "Rapport CTR / EPC ref." (qui contient "EPC ref" sans être une part) et
# "Part CTR réf. – calcul direct sur 49 régions" (accentué, variante
# d'illustration) — ni l'une ni l'autre ne doit être sélectionnée.
BUDGET_SHARE_COLUMN_PATTERNS = {
    "EPC": {"ref": r"^Part\s+EPC\s+ref", "min": r"^EPC\s+min$", "max": r"^EPC\s+max$"},
    "CTR": {"ref": r"^Part\s+CTR\s+ref", "min": r"^CTR\s+min$", "max": r"^CTR\s+max$"},
}

# Code région du Monde dans budget_shares.xlsx. Sa part vaut 1 sans encadrement
# min/max : sa sensibilité est calculée par habitant, en faisant varier la période
# de population (SHARING_PRINCIPLE_POPULATION_ROW) -- voir
# processing.compute_world_period_budget.
WORLD_REGION_CODE = "W"

# Année de population des pressions mondiales (dossier 2019_W) : la limite par
# habitant d'une période donnée leur est comparée en la ramenant à cette population.
WORLD_PRESSURE_POPULATION_ROW = 2019

# Code région utilisé pour un scénario dont le nom de dossier ne contient aucun
# code EXIOBASE (Base_year, TREND, Tech_NZE, Sufficiency_NZE... : tous des
# scénarios France).
DEFAULT_REGION_CODE = "FR"

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

# Plancher social (DLS) : partagé par plot_radial_synthesis/plot_stacked_bar.
# Gradation de bleu, du plus clair (DLS lower bound) au plus foncé (DLS upper
# bound), autour du bleu ciel historique du DLS central.
DLS_COLOR = "#00d5ff"
DLS_LOWER_COLOR = "#20a4be"
DLS_UPPER_COLOR = "#0077b6"

# Groupe de scénarios "année de base" : DLS lus sur les lignes "DLS ..." de la
# feuille DLS ; pour tout autre groupe (scénarios 2050), sur "DLS 2050 ...".
DLS_BASE_YEAR_GROUP = "Base_year"

# (clé, suffixe de ligne dans la feuille DLS, couleur, libellé de légende)
DLS_LEVELS = (
    ("dls_lower", " lower bound", DLS_LOWER_COLOR, "DLS lower bound"),
    ("dls", "", DLS_COLOR, "DLS"),
    ("dls_upper", " upper bound", DLS_UPPER_COLOR, "DLS upper bound"),
)

# Repère (insensible à la casse) du scénario tendanciel dans les noms de
# dossier ("TEND_2050", "TEND") : placé juste à côté de la référence dans les
# figures qui trient les scénarios (processing.sort_scenarios_by_reduction).
TREND_SCENARIO_MARKER = "TEND"

# ============================================================================
# FIGURE OVERSHOOT / SAFE OPERATING SPACE
# ============================================================================

# Bornes relatives (en multiples de la limite basse) du dégradé de fond safe/risk.
SAFE_LIMIT_REL = 1.0
RISK_TRANSITION_REL = 2.0
