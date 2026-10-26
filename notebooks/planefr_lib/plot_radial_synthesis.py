"""
Figure "roue radiale des limites planétaires" : 1 secteur (wedge) par LP, avec
1 barre radiale par scénario (hauteur = valeur / Safe Limit, donc le Safe
Limit est un cercle de rayon constant = 1L), une bande extérieure annotée
(1 sous-bande par scénario) et un nom de LP + unité par secteur.

Une seule fonction, pilotée par 2 booléens indépendants show_cba/show_pba
(les deux valent True par défaut -- la figure combine alors les deux) :
  - show_cba=True : valeurs consommation. Les données domestique/importé par
    catégorie existent pour le CBA -> chaque barre peut être empilée comme
    create_stacked_bar_chart (dom plein + imp hachuré, par catégorie de
    consommation), juste pliée en radial au lieu de cartésien. show_categories
    et show_import_split (bool, indépendants) activent/désactivent chacune de
    ces deux distinctions ; si les deux sont False, la barre redevient pleine,
    colorée par la couleur du LP (nuancée par scénario si show_pba=False,
    voir plus bas).
  - show_pba=True : valeurs production (payload["pba"]). C'est déjà un
    scalaire unique dans les données sources (process_single_subprocess_scenario
    ne le ventile ni par catégorie ni par dom/imp) -> chaque barre reste pleine,
    colorée par une nuance plus claire de la couleur du LP (pour la distinguer
    du CBA quand les deux sont affichés), quels que soient show_categories/
    show_import_split (ignorés côté PBA).
  - show_cba=True ET show_pba=True : les deux comptabilités sur la même
    figure -- pour chaque scénario, 2 barres adjacentes (CBA puis PBA) et 2
    sous-bandes adjacentes dans la bande extérieure.

Titre et légende s'adaptent automatiquement à show_cba/show_pba (pas besoin de
les répéter à chaque appel).

Couleurs des barres/bande extérieure : quand CBA et PBA sont affichés
ensemble (show_cba ET show_pba), chaque LP garde ses deux nuances -- pleine
pour le CBA, claire pour le PBA -- afin de distinguer les deux comptabilités.
Quand une seule des deux est affichée (figure "solo"), les deux nuances ne
servent plus à distinguer CBA/PBA (il n'y en a qu'un) : elles distinguent
alors le scénario de référence (base_year, nuance pleine -- l'ancienne
couleur "CBA") des autres scénarios (nuance claire -- l'ancienne couleur
"PBA"), qui partagent tous la même couleur. Les scénarios ne sont sinon
distingués que par leur position angulaire au sein du secteur et par le texte
de la bande extérieure -- pour les deux LP "opposés" sur la roue (Raw
material consumption / Blue water consumption), le nom du scénario est en
plus rappelé au-dessus de chaque case de la bande extérieure.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, config, io, processing
from .plot_stacked_bar import _build_category_legend

# ============================================================================
# COULEURS
# ============================================================================

LP_COLORS = {
    "N surplus":                 "#DC8968",
    "P surplus":                 "#6D4C41",  
    "Blue water consumption":    "#096DB4",  
    "Agricultural land":               "#3B5A17", 
    "GHG emissions":              "#F57C00",
    "Biodiversity loss - PDF":          "#B12BB1",
    "Biodiversity loss - HANPP":          "#FC85FC", 
    "Raw material consumption":   "#757575", 
}
_FALLBACK_LP_COLOR = "#4c78a8"

LB_COLOR = "#0d9a33"    # identique à plot_stacked_bar.py / plot_overshoot.py ("Safe Limit")
UB_COLOR = "#bc270a"    # identique à plot_stacked_bar.py / plot_overshoot.py ("Upper Safe Bound")
LOWER_COLOR = config.LOWER_SAFE_BOUND_COLOR  # identique à plot_stacked_bar.py / plot_overshoot.py ("Lower Safe Bound")

# Un fort dépassement (ex. GHG emissions à ~16L) écraserait l'échelle radiale
# de tous les secteurs : avec scale_diff=True, le plafond radial est borné à
# ce multiple de la Safe Limit, quel que soit radial_power (sauf r_max
# explicite). Les barres qui le dépassent sont tronquées au cercle
# périphérique et leur niveau réel ("16L") est écrit à leur sommet.
SCALE_DIFF_CAP_REL = 6.0

# Marge (fraction de la valeur max relative -- barre ou seuil UB/LB/DLS) ajoutée
# au-dessus de cette valeur pour fixer le plafond radial : la bande extérieure
# démarre juste au-dessus de l'élément le plus haut, sans arrondir au L entier
# supérieur (qui pouvait laisser jusqu'à 1L d'espace mort).
RADIAL_AXIS_MARGIN_FRAC = 0.02

# Pas de sous-titre récapitulant l'ordre des scénarios : pour ces deux LP,
# choisis à l'opposé l'un de l'autre sur la roue, le nom du scénario est
# rappelé directement au-dessus de la case (total/variation) qui lui
# correspond dans la bande extérieure.
SCENARIO_LABEL_SUBPROCESSES = ("Raw material consumption", "Blue water consumption")

# scale_UB=True : position commune (en multiples de la Safe Limit, avant
# radial_power) où l'Upper Safe Bound de chaque secteur est ramenée -- même
# rapport que la frontière "Increasing Risk" / "High-Risk Zone" de
# plot_overshoot (config.RISK_TRANSITION_REL / config.SAFE_LIMIT_REL).
UB_SCALED_REL = config.RISK_TRANSITION_REL / config.SAFE_LIMIT_REL


# ============================================================================
# GÉOMÉTRIE POLAIRE
# ============================================================================


def _wedge_geometry(n_lp, n_scenarios, gap_frac=0.10, bar_frac=0.82):
    """Découpe le cercle en n_lp secteurs égaux (démarrant à 12h, sens horaire),
    chacun laissant un espace visuel (gap_frac) avant de répartir n_scenarios
    barres à largeur égale (bar_frac de la place disponible par barre).

    Calcule aussi une géométrie séparée pour la bande extérieure (ring_*) :
    contrairement aux barres, la bande ne doit laisser aucun vide -- ses
    segments pavent tout le secteur (wedge_span entier, pas usable_span) et
    se touchent d'un secteur à l'autre.

    Returns:
        dict avec wedge_span/usable_span (rad), boundaries (limites de secteur),
        starts (début de la zone utile de chaque secteur), centers (liste de
        listes : centers[lp_idx][scenario_idx]), bar_width, ring_centers (même
        structure que centers, mais pavant tout wedge_span sans espace),
        ring_width -- tous les angles en RADIANS (ax.bar en polaire attend
        width en radians, pas en degrés).
    """
    wedge_span = 2 * np.pi / n_lp
    gap = wedge_span * gap_frac
    usable_span = wedge_span - gap
    bar_slot = usable_span / n_scenarios
    bar_width = bar_slot * bar_frac

    boundaries = [i * wedge_span for i in range(n_lp)]
    starts = [b + gap / 2 for b in boundaries]
    centers = [[s + bar_slot * (k + 0.5) for k in range(n_scenarios)] for s in starts]

    ring_width = wedge_span / n_scenarios
    ring_centers = [[b + ring_width * (k + 0.5) for k in range(n_scenarios)] for b in boundaries]

    return dict(wedge_span=wedge_span, usable_span=usable_span, boundaries=boundaries,
                starts=starts, centers=centers, bar_width=bar_width,
                ring_centers=ring_centers, ring_width=ring_width)


def _ub_slope(ub_rel):
    """Pente de l'échelle scale_UB au-delà de la Safe Limit : amène l'Upper Safe
    Bound (ub_rel, en multiples de la Safe Limit) sur UB_SCALED_REL. None si la
    ligne n'a pas d'Upper Safe Bound exploitable (absente ou <= Safe Limit) --
    le secteur garde alors l'échelle 1L."""
    if ub_rel is None or not np.isfinite(ub_rel) or ub_rel <= 1:
        return None
    return (UB_SCALED_REL - 1) / (ub_rel - 1)


def _scale_rel(rel, slope):
    """Échelle propre à un secteur (scale_UB, même logique que
    plot_overshoot.create_overshoot_safe_space_figure_by_region) : linéaire
    par morceaux, inchangée entre 0 et la Safe Limit (1L), puis de pente
    slope au-delà (voir _ub_slope). Identité si slope est None."""
    if rel is None or slope is None or rel <= 1:
        return rel
    return 1 + (rel - 1) * slope


def _format_tick(t):
    """Libellé "NL" d'une graduation (entier ou fractionnaire)."""
    return f"{t:g}L"


def _radial_rotation_deg(theta_rad):
    """Rotation (degrés écran) pour un texte aligné le long d'un rayon (ex. les
    graduations "NL"), toujours lisible (jamais à l'envers). Suppose
    set_theta_zero_location('N') + set_theta_direction(-1)."""
    deg = (90.0 - np.degrees(theta_rad)) % 360
    if 90 < deg < 270:
        deg = (deg + 180) % 360
    return deg


def _tangential_rotation_deg(theta_rad):
    """Rotation (degrés écran) pour un texte aligné tangentiellement (le long du
    cercle, ex. les noms de LP) -- orthogonale à _radial_rotation_deg."""
    deg = (-np.degrees(theta_rad)) % 360
    if 90 < deg < 270:
        deg = (deg + 180) % 360
    return deg


def _nice_radial_ticks(rel_cap, max_ticks=6):
    """Graduations entières (en multiples de la Safe Limit) de 1 à floor(rel_cap)
    (le plafond radial réel étant fixé séparément, juste au-dessus de rel_cap --
    voir RADIAL_AXIS_MARGIN_FRAC -- une graduation à rel_cap lui-même n'aurait
    pas de sens si ce n'est pas un entier), en sous-échantillonnant si trop
    nombreuses (même idée que plot_overshoot._set_break_axis_ticks). 1L est
    toujours inclus (cercle Safe Limit)."""
    r_floor = max(1, int(np.floor(rel_cap)))
    ticks = list(range(1, r_floor + 1))
    if len(ticks) > max_ticks:
        step = max(1, int(np.ceil(r_floor / max_ticks)))
        ticks = list(range(step, r_floor + 1, step))
        if ticks[-1] != r_floor:
            ticks.append(r_floor)
    return sorted(set(ticks) | {1})


# ============================================================================
# EXTRACTION DES VALEURS, EMPILEMENT PAR CATÉGORIE, ANNOTATION
# ============================================================================


def _stack_segments(payload, categories, threshold_ref, group_by_category,
                     show_categories, show_import_split, fallback_color):
    """Décompose l'empreinte CBA d'un (LP, scénario) en segments empilables, en
    unités relatives à threshold_ref (Safe Limit), prêts à dessiner (couleur déjà résolue). to_r() doit être appliqué aux BORNES
    cumulées par l'appelant (pas à la hauteur du segment), seule façon de
    rester correct en échelle non-linéaire (radial_power != 1).

    show_categories/show_import_split pilotent l'habillage, indépendamment :
      - les deux à True  : 1 segment par (catégorie, dom/imp), coloré par
        catégorie (comme create_stacked_bar_chart), hachuré si importé.
      - show_categories=True, show_import_split=False : 1 segment par
        catégorie (dom+imp fusionnés), coloré par catégorie, jamais hachuré.
      - show_categories=False, show_import_split=True : 2 segments (tout
        domestique, tout importé), colorés par fallback_color (couleur du LP),
        le 2e hachuré.
      - les deux à False : 1 seul segment (total), coloré par fallback_color --
        identique visuellement à une barre PBA.

    Returns:
        liste de (rel_bottom, rel_top, color, is_import), dans l'ordre
        d'empilement (dom puis imp par groupe si group_by_category, sinon tout
        dom puis tout imp -- non pertinent si show_import_split=False).
    """
    if payload is None or threshold_ref is None:
        return []

    def value_for(key, category):
        return payload[key].get(category, 0) if key in payload else 0

    groups = list(enumerate(categories)) if show_categories else [(0, None)]

    def dom_imp(category):
        if category is not None:
            return value_for("domestique", category), value_for("importé", category)
        return (sum(value_for("domestique", c) for c in categories),
                sum(value_for("importé", c) for c in categories))

    entries = [(cat_idx, category, *dom_imp(category)) for cat_idx, category in groups]

    order = []  # (cat_idx, category, valeur_absolue, is_import)
    if not show_import_split:
        for cat_idx, category, dom_val, imp_val in entries:
            order.append((cat_idx, category, dom_val + imp_val, False))
    elif group_by_category:
        for cat_idx, category, dom_val, imp_val in entries:
            order.append((cat_idx, category, dom_val, False))
            order.append((cat_idx, category, imp_val, True))
    else:
        for cat_idx, category, dom_val, imp_val in entries:
            order.append((cat_idx, category, dom_val, False))
        for cat_idx, category, dom_val, imp_val in entries:
            order.append((cat_idx, category, imp_val, True))

    segments = []
    cumulative = 0.0
    for cat_idx, category, abs_val, is_import in order:
        rel_val = abs_val / threshold_ref
        if rel_val > 0:
            color = colors.get_dark_shade(colors.get_category_color(category, cat_idx)) if show_categories else fallback_color
            segments.append((cumulative, cumulative + rel_val, color, is_import))
        cumulative += rel_val
    return segments


def _annotate(val, ref_val, is_reference, is_pba=False):
    """Même règle de texte que create_synthesis_figure_by_lp : valeur absolue en
    gras noir pour le scénario de référence, variation % colorée (rouge = hausse,
    vert = baisse) pour les autres. PBA utilise un rouge/vert plein (plus saturé
    que le lightcoral/lightgreen du CBA) pour rester lisible malgré la nuance
    claire de la barre PBA sous-jacente."""
    if is_reference or ref_val is None or ref_val <= 0:
        return f"{val:.0f}", "black"
    variation_pct = (val - ref_val) / ref_val * 100
    text = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
    if is_pba:
        color = "#991616" if variation_pct > 0 else ("#237A23" if variation_pct < 0 else "black")
    else:
        color = "#F34E4B" if variation_pct > 0 else ("#7ADB67" if variation_pct < 0 else "black")
    return text, color


def _bdf_column(bdf_df, subprocess_name):
    """Colonne de la feuille Bdf correspondant au sous-processus (nommée
    "<sous-processus> (<unité>)"), ou None si absente."""
    for col in bdf_df.columns:
        if str(col).split(" (")[0].strip() == subprocess_name:
            return col
    return None


# ============================================================================
# FIGURE PRINCIPALE
# ============================================================================


def create_radial_synthesis_figure(
    all_scenarios_data, seuils_df, scenario_names,
    show_cba=True, show_pba=True,
    dls_df=None, show_dls=False,
    radial_power=0.5,
    show_categories=True, show_import_split=True, group_by_category=False,
    lp_colors=None,
    r_max=None,
    display_bounds=True,
    title=None,
    figsize=(16, 16),
    sharing_principle=None, pop_df=None,
    scale_UB=False, scale_diff=False,
    scenario_group=None,
    Bdf=None,
):
    """Roue radiale : 1 secteur par LP, 1 barre par scénario, hauteur = valeur /
    Safe Limit. Le cercle Safe Limit (vert, seuil historique config.THRESHOLD_LB_ABS)
    est donc à rayon constant = 1L sur toute la figure ; l'arc Upper Safe Bound
    (rouge, config.THRESHOLD_UB_ABS) et l'arc Lower Safe Bound (vert clair,
    config.THRESHOLD_LOWER_ABS, toujours en-deçà du Safe Limit) sont propres à
    chaque secteur (leur ratio au Safe Limit varie selon le LP) -- ces 3 noms
    reprennent les lignes "brutes" de seuils.xlsx/feuille "Synthèse" ("Lower
    safe bound"/"Safe limit"/"Upper safe bound"). display_bounds=False masque
    Lower Safe Bound et Upper Safe Bound (seul le cercle Safe Limit reste
    affiché). Bande extérieure : reprend l'annotation de
    create_synthesis_figure_by_lp (valeur absolue pour le scénario de
    référence, variation % pour les autres), repositionnée en 1 sous-bande par
    scénario.

    Args:
        all_scenarios_data: liste de {lp_name: résultat de process_scenario},
            un élément par scénario, dans l'ordre de scenario_names.
        show_cba, show_pba: quelles comptabilités afficher (au moins une des
            deux doit être True) :
            - show_cba seul : barre empilable, voir show_categories/
              show_import_split.
            - show_pba seul : barre pleine (une seule valeur scalaire existe
              dans les données sources, jamais de ventilation par
              catégorie/origine possible), nuance claire de la couleur du LP.
            - les deux True (défaut) : côte à côte -- pour chaque scénario,
              2 barres adjacentes (CBA pleine/empilée, PBA en nuance claire)
              et la bande extérieure suit le même découpage, 2 sous-bandes par
              scénario. Titre et légende s'adaptent automatiquement.
            - une seule des deux affichée (figure "solo") : les nuances
              pleine/claire ne distinguent plus CBA/PBA (redondant, il n'y en
              a qu'un) mais le scénario de référence (base_year, nuance
              pleine) des autres scénarios (nuance claire), pour toute case où
              show_categories ne prend pas déjà le dessus (voir ci-dessous).
        show_categories: (CBA uniquement) True = empile 1 segment par catégorie
            de consommation, coloré comme create_stacked_bar_chart (la
            distinction base_year/autres scénarios reste visible dans la
            bande extérieure, mais plus dans les segments eux-mêmes). False =
            pas de distinction de catégorie ; la barre prend alors la couleur
            du LP, nuancée par scénario comme décrit plus haut.
        show_import_split: (CBA uniquement) True = distingue domestique (plein)
            / importé (hachuré). False = domestique + importé fusionnés dans le
            même segment.
        group_by_category: voir create_stacked_bar_chart -- pertinent seulement
            si show_categories=True et show_import_split=True (ordre
            d'empilement dom/imp entrelacé par catégorie, ou tout dom puis tout
            imp).
        show_dls, dls_df: si show_dls=True, ajoute par secteur les arcs DLS
            (plancher social), DLS lower bound et DLS upper bound (gradation de
            bleu, clair -> foncé), quand dls_df fournit une valeur pour ce LP :
            valeurs nationales lues telles quelles dans la feuille DLS (voir
            processing.lookup_dls). Lignes "DLS ..."
            si scenario_group vaut DLS_BASE_YEAR_GROUP ("Base_year"), "DLS 2050
            ..." sinon.
        scenario_group: groupe de scénarios affiché (config.get_scenario_folders),
            sert uniquement à choisir les lignes DLS (voir show_dls).
        radial_power: exposant de la transformation rayon = (valeur/LB)**radial_power.
            1.0 = échelle linéaire stricte. 0.5 (défaut) = racine carrée. Plus
            la valeur est petite (ex. 1/3), plus l'échelle est "dilatée" : les
            LP en-deçà de leur limite (ex. Blue water consumption, <1L) sont
            étirés vers l'extérieur et donc plus visibles, tandis que les LP en
            fort dépassement (ex. GHG emissions) sont d'autant plus compressés
            -- même principe que sqrt, en plus prononcé. Le cercle Safe Limit
            reste à un rayon constant (=1) quelle que soit la valeur, car 1
            élevé à n'importe quelle puissance vaut toujours 1.
            Quelle que soit sa valeur, avec scale_diff=True le plafond radial
            est borné à SCALE_DIFF_CAP_REL (6L) -- voir r_max.
        lp_colors: dict optionnel pour surcharger/étendre LP_COLORS.
        r_max: plafond optionnel de l'axe radial, en multiples du Safe Limit
            (même unité que les graduations "NL") ; None = auto-calculé depuis
            les données (valeurs + seuils affichés -- Upper/Lower Safe Bound
            si display_bounds=True, DLS si show_dls=True), borné à
            SCALE_DIFF_CAP_REL si scale_diff=True. Avec scale_diff=True, toute
            barre qui dépasse le plafond est tronquée au cercle périphérique,
            avec son niveau réel (ex. "16L") écrit à son sommet ; les arcs de
            seuil au-delà du plafond ne sont pas tracés.
        display_bounds: True (défaut) affiche les arcs Lower Safe Bound et
            Upper Safe Bound en plus du cercle Safe Limit (toujours affiché).
            False les masque (et ne les compte plus dans le calcul de r_max).
        title: préfixe de titre optionnel (ex. "Planetary boundaries assessment
            of French NZE Scenarios") ; le suffixe (" - Consumption-based",
            " - Production-based" ou " - CBA vs PBA") est ajouté automatiquement
            selon show_cba/show_pba. Sans préfixe, seul le suffixe est affiché.
        sharing_principle: si fourni ("Equality 2050"/"Equality 2100"/"Equality
            2019"), threshold_lb/threshold_ub (LP) sont recalculés à la volée
            par un partage égal per capita du budget mondial pour la période de
            référence associée (voir config.SHARING_PRINCIPLE_POPULATION_ROW et
            processing.compute_sharing_seuil), au lieu d'être lus statiquement
            dans seuils_df. Nécessite pop_df.
        pop_df: feuille "Population" de seuils.xlsx (io.load_population_df) --
            requis seulement si sharing_principle est fourni.
        scale_UB: si False (défaut), échelle commune à tous les secteurs, en
            multiples de la Safe Limit (1L, 2L, 3L...). Si True (même principe
            que plot_overshoot.create_overshoot_safe_space_figure_by_region),
            l'échelle de chaque secteur est propre à son LP, linéaire par
            morceaux (avant radial_power) : inchangée entre 0 et la Safe Limit
            (1L), puis étirée ou comprimée pour que l'Upper Safe Bound tombe
            exactement sur UB_SCALED_REL (2L) ; au-delà, la pente du segment
            [Safe Limit, UB] est prolongée. La figure montre alors deux cercles
            concentriques continus : Safe Limit (vert) et Upper Safe Bound
            (rouge, si display_bounds=True). Barres, Lower Safe Bound et DLS
            suivent cette échelle. Les graduations "NL" au-delà de 1L, qui
            n'ont plus de sens commun, sont masquées. Un secteur sans Upper
            Safe Bound (ou avec UB <= Safe Limit) garde l'échelle 1L.
        scale_diff: False (défaut) = affichage classique, échelle commune
            jusqu'au point le plus haut de la figure. True = plafond radial
            borné à SCALE_DIFF_CAP_REL (6L) : les barres qui le dépassent vont
            jusqu'au cercle périphérique et leur niveau réel (ex. "16L") est
            écrit juste en dessous (voir r_max). Incompatible avec scale_UB.
        Bdf: None (défaut) = figure multi-scénarios classique. Sinon, nom d'une
            feuille de config.BDF_FILE (ex. "décile") : les scénarios sont
            remplacés par la valeur agrégée (ligne "Total", en 1re position,
            nuance pleine) puis les catégories de ménages de la 1re colonne de
            cette feuille (nuance claire), 1 barre pleine chacune, de
            hauteur = pression de la catégorie / (Safe Limit nationale x
            weight) -- chaque catégorie reçoit la part "weight" du budget
            national. Upper/Lower Safe Bound et DLS étant pondérés par le même
            weight, leur rapport au Safe Limit est inchangé : les arcs restent
            communs à tout le secteur. La bande extérieure affiche la valeur
            absolue pour le Total et, pour chaque catégorie, la variation
            relative de son overshoot (hauteur de barre) par rapport à celui
            du Total (ex. +20%). show_cba/show_pba,
            show_categories/show_import_split et le tri des scénarios sont
            ignorés ; all_scenarios_data ne sert qu'à fixer la liste/l'ordre
            des LP (scénario de référence).

    Returns:
        (fig, ax)
    """
    if not show_cba and not show_pba:
        raise ValueError("show_cba et show_pba ne peuvent pas être tous les deux False.")
    if scale_UB and scale_diff:
        raise ValueError("scale_UB et scale_diff ne peuvent pas être tous les deux True.")
    is_bdf = Bdf is not None
    if is_bdf:
        show_cba, show_pba = True, False
    combined = show_cba and show_pba

    reference_idx = config.REFERENCE_SCENARIO_IDX
    if not is_bdf:
        # Le scénario de référence (base_year) garde sa position, suivi du
        # tendanciel (config.TREND_SCENARIO_MARKER), puis des autres scénarios par ordre
        # décroissant de variation moyenne de l'empreinte vs la référence (voir
        # processing.mean_variation) : la plus faible baisse / plus forte hausse juste à côté
        # du tendanciel, la plus forte baisse le plus loin.
        value_keys = (["cba"] if show_cba else []) + (["pba"] if show_pba else [])
        display_order = processing.sort_scenarios_by_reduction(all_scenarios_data, scenario_names, reference_idx, value_keys)
        scenario_names = [scenario_names[i] for i in display_order]
        all_scenarios_data = [all_scenarios_data[i] for i in display_order]

    first_scenario_data = all_scenarios_data[reference_idx]
    subprocesses = list(first_scenario_data.keys())

    if is_bdf:
        # Les "scénarios" deviennent la valeur agrégée (ligne "Total", en
        # 1re position = référence) suivie des catégories de ménages.
        bdf_df = io.load_bdf(Bdf)
        bdf_df = pd.concat([bdf_df.loc[[config.BDF_TOTAL_ROW]],
                            bdf_df.drop(index=config.BDF_TOTAL_ROW)])
        bdf_weights = bdf_df[config.BDF_WEIGHT_COL].astype(float).tolist()
        scenario_names = list(bdf_df.index)
        reference_idx = 0

    n_lp = len(subprocesses)
    n_scenarios = len(scenario_names)
    # En mode combiné, chaque scénario occupe 2 emplacements adjacents (CBA
    # puis PBA) : on construit la géométrie pour 2×n_scenarios "créneaux"
    # virtuels, et on retrouve le scénario réel via slot_idx // 2.
    n_slots = n_scenarios * 2 if combined else n_scenarios

    palette = dict(LP_COLORS)
    if lp_colors:
        palette.update(lp_colors)

    def to_r(rel):
        return float(rel) ** radial_power

    geo = _wedge_geometry(n_lp, n_slots)

    # PREMIÈRE PASSE : valeurs, seuils, unité et catégories par LP (et rayon
    # max nécessaire pour englober valeurs + seuils UB/DLS affichés)
    lp_info = []
    legend_categories = set()
    max_rel = 1.0
    for subprocess_name in subprocesses:
        threshold_lb = processing.lookup_threshold(seuils_df, config.THRESHOLD_LB_ABS, subprocess_name,
                                                    pop_df=pop_df, sharing_principle=sharing_principle)
        threshold_ub = processing.lookup_threshold(seuils_df, config.THRESHOLD_UB_ABS, subprocess_name,
                                                    pop_df=pop_df, sharing_principle=sharing_principle)
        threshold_lower = processing.lookup_threshold(seuils_df, config.THRESHOLD_LOWER_ABS, subprocess_name,
                                                        pop_df=pop_df, sharing_principle=sharing_principle)
        unit_text = processing.lookup_first_text(seuils_df, config.UNIT_ROW_ABS, subprocess_name)
        # DLS (et ses bornes), par clé de config.DLS_LEVELS ; None si absent pour ce LP
        dls_values = {key: None for key, _, _, _ in config.DLS_LEVELS}
        if show_dls and dls_df is not None:
            dls_prefix = "DLS" if scenario_group == config.DLS_BASE_YEAR_GROUP else "DLS 2050"
            for key, suffix, _, _ in config.DLS_LEVELS:
                dls_values[key] = processing.lookup_dls(dls_df, dls_prefix + suffix, subprocess_name)

        categories = (first_scenario_data.get(subprocess_name) or {}).get("categories", [])
        legend_categories.update(categories)

        if is_bdf:
            # 1 valeur par catégorie de ménages (Total en tête).
            bdf_col = _bdf_column(bdf_df, subprocess_name)
            values = [float(v) for v in bdf_df[bdf_col]] if bdf_col is not None else [None] * n_scenarios
            # Seuils propres à chaque catégorie : seuil national x weight.
            weights = bdf_weights
            scale_values = values
        else:
            cba_values = [processing.extract_value(sd.get(subprocess_name), "cba") for sd in all_scenarios_data]
            pba_values = [processing.extract_value(sd.get(subprocess_name), "pba") for sd in all_scenarios_data]
            if combined:
                values = []
                for cba_val, pba_val in zip(cba_values, pba_values):
                    values.append(cba_val)
                    values.append(pba_val)
            else:
                values = cba_values if show_cba else pba_values
            weights = [1.0] * n_slots
            # L'échelle radiale (graduations "NL", plafond) est toujours calée sur
            # le max CBA/PBA, même en solo (show_cba seul ou show_pba seul) : une
            # figure PBA seule doit avoir la même graduation qu'une figure CBA
            # seule ou combinée pour le même secteur, pas une graduation resserrée
            # sur ses seules valeurs affichées.
            scale_values = cba_values + pba_values

        threshold_ref = threshold_lb
        if threshold_ref is not None:
            rel_values = [(v / (threshold_ref * w)) if v is not None else None for v, w in zip(values, weights)]
            scale_rel_values = (rel_values if is_bdf else
                                [(v / threshold_ref) if v is not None else None for v in scale_values])
            ub_rel = (threshold_ub / threshold_ref) if threshold_ub is not None else None
            lb_rel = (threshold_lb / threshold_ref) if threshold_lb is not None else None
            lower_rel = (threshold_lower / threshold_ref) if threshold_lower is not None else None
            dls_rels = {key: (v / threshold_ref) if v is not None else None for key, v in dls_values.items()}
        else:
            rel_values = [None] * n_slots
            scale_rel_values = []
            ub_rel = None
            lb_rel = None
            lower_rel = None
            dls_rels = {key: None for key in dls_values}

        # scale_UB : pente propre au secteur (None = échelle 1L inchangée). Les
        # valeurs relatives restent stockées non remises à l'échelle ;
        # _scale_rel est appliqué au moment du tracé (voir r_of).
        slope = _ub_slope(ub_rel) if scale_UB else None

        bound_rels = [ub_rel, lower_rel] if display_bounds else []
        for rel in scale_rel_values + [lb_rel, *dls_rels.values()] + bound_rels:
            if rel is not None and np.isfinite(rel):
                max_rel = max(max_rel, _scale_rel(rel, slope))

        lp_info.append(dict(
            name=subprocess_name, unit=unit_text, threshold_ref=threshold_ref, categories=categories,
            values=values, rel_values=rel_values, ub_rel=ub_rel, lb_rel=lb_rel, lower_rel=lower_rel,
            dls_rels=dls_rels, slope=slope, ub_slope=slope,
            color=palette.get(subprocess_name, _FALLBACK_LP_COLOR),
        ))

    if r_max is not None:
        rel_cap_input = r_max
    elif scale_diff:
        rel_cap_input = min(max_rel, SCALE_DIFF_CAP_REL)
    else:
        rel_cap_input = max_rel

    def is_clipped(info, rel):
        """True si la valeur relative dépasse le plafond radial (rel_cap)."""
        return rel is not None and _scale_rel(rel, info["slope"]) > rel_cap

    def r_of(info, rel):
        """Rayon tracé d'une valeur relative du secteur info : remise à
        l'échelle scale_UB (identité sinon), tronquée au plafond radial
        (cercle périphérique), puis radial_power."""
        return to_r(min(_scale_rel(rel, info["slope"]), rel_cap))

    # Le plafond radial est fixé juste au-dessus (RADIAL_AXIS_MARGIN_FRAC) de la
    # valeur relative la plus haute (barre ou seuil UB/LB/DLS affiché), pour que
    # la bande extérieure démarre tout près du sommet réel -- pas de l'arrondi
    # au L entier supérieur, qui pouvait laisser jusqu'à 1L d'espace mort. Les
    # graduations "NL", elles, restent des multiples entiers (jusqu'au dernier
    # ≤ ce sommet réel).
    # scale_UB : seule la graduation 1L reste commune à tous les secteurs.
    rel_ticks = [1] if scale_UB else _nice_radial_ticks(rel_cap_input)
    rel_cap = rel_cap_input * (1 + RADIAL_AXIS_MARGIN_FRAC)
    r_max_plot = to_r(rel_cap)
    ring_thickness = r_max_plot * 0.11
    r_ring_start = r_max_plot
    r_ring_end = r_ring_start + ring_thickness
    r_label = r_max_plot * 0.78
    # Rayon des noms de scénario rappelés au-dessus des cases de la bande
    # extérieure (SCENARIO_LABEL_SUBPROCESSES uniquement) -- juste à
    # l'extérieur de la bande, voir _nice_radial_ticks/set_rlim plus bas.
    r_scenario_label = r_ring_end + ring_thickness * 0.7
    has_scenario_labels = any(name in SCENARIO_LABEL_SUBPROCESSES for name in subprocesses)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    # Marges par défaut de matplotlib très généreuses pour un unique subplot
    # (~12% de chaque côté) : on les resserre pour que le cercle remplisse la
    # figure, avec juste la place nécessaire au-dessus pour le titre et
    # en-dessous pour la légende (positionnés juste après, au ras de ces
    # nouvelles bornes).
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.058)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # Grille de fond : cercles concentriques + spokes aux limites de secteur,
    # avec graduations "NL" répétées à chacune des n_lp limites.
    theta_full = np.linspace(0, 2 * np.pi, 400)
    for t in rel_ticks:
        if t == 1:
            continue  # tracé séparément, en gras, comme cercle Safe Limit
        ax.plot(theta_full, np.full_like(theta_full, to_r(t)), color="#e0e0e0", linewidth=0.7, zorder=1)

    for lp_idx, b in enumerate(geo["boundaries"]):
        ax.plot([b, b], [0, r_ring_end], color="#bdbdbd", linewidth=0.8, zorder=1)
        offset = geo["wedge_span"] * 0.02
        for t in rel_ticks:
            ax.text(b + offset, to_r(t), _format_tick(t), ha="left", va="center", fontsize=12,
                     color="#555555", rotation=_radial_rotation_deg(b), rotation_mode="anchor",
                     bbox=dict(facecolor="white", edgecolor="white", alpha=0.7, pad=0.5), zorder=3)

    # DEUXIÈME PASSE : barres, bande extérieure, arcs UB/DLS, nom de LP
    for lp_idx, info in enumerate(lp_info):
        centers = geo["centers"][lp_idx]
        bar_width = geo["bar_width"]
        ring_centers = geo["ring_centers"][lp_idx]

        def _bar_hatch(is_import=False):
            # Hachurage "/" pour l'importé uniquement (le scénario de
            # référence n'est plus hachuré, voir _solo_color).
            return "/" if is_import else None

        # PBA : toujours en nuance plus claire de la couleur du LP.
        pba_color = colors.get_light_shade(info["color"], 1.5)

        def _solo_color(scenario_idx):
            # Hors mode combiné, CBA et PBA partagent désormais une seule
            # couleur par LP (celle du PBA) : les deux nuances ne servent
            # plus à distinguer CBA/PBA (il n'y en a qu'un affiché) mais le
            # scénario de référence (base_year, nuance pleine -- l'ancienne
            # couleur "CBA") des autres scénarios (nuance claire), à la place
            # du hachurage "xx" utilisé auparavant. Bdf : la valeur agrégée
            # (Total) joue le rôle du scénario de référence.
            return info["color"] if scenario_idx == reference_idx else pba_color

        def _draw_cba_bar(theta, scenario_idx):
            payload = all_scenarios_data[scenario_idx].get(info["name"])
            fallback_color = info["color"] if combined else _solo_color(scenario_idx)
            segments = _stack_segments(payload, info["categories"], info["threshold_ref"], group_by_category,
                                        show_categories, show_import_split, fallback_color)
            for rel_bottom, rel_top, color, is_import in segments:
                ax.bar([theta], [r_of(info, rel_top) - r_of(info, rel_bottom)], width=bar_width,
                       bottom=r_of(info, rel_bottom),
                       color=color, edgecolor="white", linewidth=0.4,
                       hatch=_bar_hatch(is_import), zorder=5)

        if combined:
            # 2 barres adjacentes par scénario -- CBA (empilée comme
            # _draw_cba_bar) puis PBA (toujours pleine, nuance claire).
            for scenario_idx, theta in enumerate(centers[0::2]):
                _draw_cba_bar(theta, scenario_idx)
            pba_centers = centers[1::2]
            pba_heights = [r_of(info, rel) if rel is not None else 0.0 for rel in info["rel_values"][1::2]]
            for scenario_idx, (theta, height) in enumerate(zip(pba_centers, pba_heights)):
                ax.bar([theta], [height], width=bar_width, bottom=0.0,
                       color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        elif show_cba and not is_bdf:
            # Empreinte détaillée disponible (domestique/importé par catégorie) :
            # barre empilée selon show_categories/show_import_split (les deux à
            # False -> 1 segment plein, coloré par _solo_color). to_r() est
            # appliqué aux BORNES cumulées (pas à la hauteur du segment), seule
            # façon correcte de rester cohérent avec le cercle LB/l'arc UB en
            # échelle non-linéaire (radial_power != 1).
            for scenario_idx, theta in enumerate(centers):
                _draw_cba_bar(theta, scenario_idx)
        else:
            # PBA seul (ou Bdf) : une seule valeur scalaire par (LP, scénario) dans les
            # données sources -> pas de ventilation possible, barre pleine.
            heights = [r_of(info, rel) if rel is not None else 0.0 for rel in info["rel_values"]]
            for scenario_idx, (theta, height) in enumerate(zip(centers, heights)):
                ax.bar([theta], [height], width=bar_width, bottom=0.0,
                       color=_solo_color(scenario_idx), edgecolor="white", linewidth=0.6, zorder=5)

        # Bande extérieure : pavage complet du secteur (ring_centers/ring_width,
        # sans le gap_frac des barres) pour ne laisser aucun vide entre
        # scénarios ni entre secteurs. En mode combiné, la moitié PBA reprend
        # la nuance claire de la barre correspondante ; sinon chaque case suit
        # la couleur de la barre du scénario correspondant (_solo_color).
        if combined:
            ax.bar(ring_centers[0::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=info["color"], edgecolor="white", linewidth=0.6, zorder=5)
            ax.bar(ring_centers[1::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        else:
            ring_colors = [_solo_color(scenario_idx) for scenario_idx in range(n_scenarios)]
            ax.bar(ring_centers, [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=ring_colors, edgecolor="white", linewidth=0.6, zorder=5)

        # Texte de la bande : valeur/scénario de référence en fonction du
        # créneau (CBA ou PBA) en mode combiné, de la seule comptabilité
        # affichée sinon.
        if combined:
            ref_cba, ref_pba = info["values"][reference_idx * 2], info["values"][reference_idx * 2 + 1]
            ref_by_slot = [ref_cba if i % 2 == 0 else ref_pba for i in range(n_slots)]
            is_ref_by_slot = [(i // 2) == reference_idx for i in range(n_slots)]
            is_pba_by_slot = [i % 2 == 1 for i in range(n_slots)]
        else:
            ref_by_slot = [info["values"][reference_idx]] * n_slots
            is_ref_by_slot = [i == reference_idx for i in range(n_slots)]
            # Une seule comptabilité affichée : la couleur des variations
            # s'aligne toujours sur le style PBA (rouge/vert saturé), que ce
            # soit le CBA ou le PBA qui soit affiché seul.
            is_pba_by_slot = [True] * n_slots

        r_ring_mid = (r_ring_start + r_ring_end) / 2
        for i, theta in enumerate(ring_centers):
            val = info["values"][i]
            if val is None:
                continue
            if is_bdf and i != reference_idx:
                # Variation relative de l'overshoot (pression / seuil pondéré)
                # de la catégorie par rapport à celui de la valeur agrégée.
                text, color = _annotate(info["rel_values"][i], info["rel_values"][reference_idx], False, True)
            else:
                text, color = _annotate(val, ref_by_slot[i], is_ref_by_slot[i], is_pba_by_slot[i])
            ax.text(theta, r_ring_mid, text, ha="center", va="center",
                    fontsize=14.5, fontweight="bold", color=color, zorder=9)

        # Nom du scénario rappelé au-dessus de la/les case(s) de la bande
        # extérieure -- seulement pour les 2 LP opposés sur la roue
        # (remplace le sous-titre "Scenario order : ..." global). En mode
        # combiné, un scénario occupe 2 cases adjacentes (CBA puis PBA) : le
        # nom est centré au-dessus des deux.
        if info["name"] in SCENARIO_LABEL_SUBPROCESSES:
            for scenario_idx in range(n_scenarios):
                if combined:
                    theta_label = (ring_centers[scenario_idx * 2] + ring_centers[scenario_idx * 2 + 1]) / 2
                else:
                    theta_label = ring_centers[scenario_idx]
                ax.text(theta_label, r_scenario_label, scenario_names[scenario_idx], ha="center", va="center",
                        fontsize=12, fontweight="bold", color="#333333",
                        rotation=_tangential_rotation_deg(theta_label), rotation_mode="anchor", zorder=9)

        # Niveau réel des barres tronquées au plafond radial (ex. "16L"), écrit
        # au milieu de leur sommet, juste sous le cercle périphérique.
        for theta, rel in zip(centers, info["rel_values"]):
            if is_clipped(info, rel):
                ax.text(theta, r_max_plot * 0.97, _format_tick(round(rel)), ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black",
                        rotation=_tangential_rotation_deg(theta), rotation_mode="anchor", zorder=9,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        if (display_bounds and info["ub_rel"] is not None and info["ub_slope"] is None
                and not is_clipped(info, info["ub_rel"])):
            # Secteur remis à l'échelle (scale_UB) : son Upper Safe Bound est
            # tracée plus bas, avec le cercle continu à UB_SCALED_REL.
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, r_of(info, info["ub_rel"])), color=UB_COLOR,
                    linewidth=3.0, solid_capstyle="butt", zorder=8)

        if display_bounds and info["lower_rel"] is not None:
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, r_of(info, info["lower_rel"])), color=LOWER_COLOR,
                    linewidth=3.0, solid_capstyle="butt", zorder=8)

        # Arcs DLS : tout le secteur utile, sauf en mode combiné où ils ne
        # couvrent que les barres CBA (le DLS est un plancher de consommation,
        # sans pertinence pour la barre PBA adjacente).
        if combined:
            dls_spans = [(theta - bar_width / 2, theta + bar_width / 2) for theta in centers[0::2]]
        else:
            dls_spans = [(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"])]
        for key, _, dls_color, _ in config.DLS_LEVELS:
            dls_rel = info["dls_rels"][key]
            if dls_rel is None or is_clipped(info, dls_rel):
                continue
            for theta_start, theta_end in dls_spans:
                theta_arc = np.linspace(theta_start, theta_end, 40)
                ax.plot(theta_arc, np.full_like(theta_arc, r_of(info, dls_rel)), color=dls_color,
                        linestyle="--", linewidth=3.0, zorder=8)

        bisector = geo["boundaries"][lp_idx] + geo["wedge_span"] / 2
        label = f"{info['name']}\n({info['unit']})" if info["unit"] else info["name"]
        txt = ax.text(bisector, r_label, label, ha="center", va="center",
                       fontsize=16.5, fontweight="bold", color=info["color"], linespacing=1.4,
                       rotation=_tangential_rotation_deg(bisector), rotation_mode="anchor", zorder=9)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Cercle Safe Limit "1L" : rayon constant = 1 (point fixe de to_r, quel que
    # soit radial_power), tracé par-dessus les barres.
    ax.plot(theta_full, np.full_like(theta_full, to_r(1.0)), color=LB_COLOR, linewidth=3.2, zorder=8)
    for lp_idx, info in enumerate(lp_info):
        theta_arc = np.linspace(geo["boundaries"][lp_idx], geo["boundaries"][lp_idx] + geo["wedge_span"], 40)
        # scale_UB : second cercle concentrique, Upper Safe Bound ramenée à
        # UB_SCALED_REL dans chaque secteur remis à l'échelle -- arcs pavant
        # tout wedge_span pour former un cercle continu.
        if display_bounds and info["ub_slope"] is not None:
            ax.plot(theta_arc, np.full_like(theta_arc, to_r(UB_SCALED_REL)), color=UB_COLOR,
                    linewidth=3.2, zorder=8)

    r_axis_max = (r_scenario_label + ring_thickness * 0.6) if has_scenario_labels else (r_ring_end * 1.02)
    ax.set_rlim(0, r_axis_max)

    # Légende : lignes de seuil communes aux deux figures, puis les catégories
    # si affichées (CBA avec show_categories=True -- explique l'habillage des
    # barres empilées) et/ou l'entrée "Import" seule si la distinction dom/imp
    # est affichée sans les catégories.
    legend_handles = [Line2D([0], [0], color=LB_COLOR, linewidth=3, label="Safe Limit")]
    if display_bounds:
        legend_handles = [
            Line2D([0], [0], color=LOWER_COLOR, linewidth=3, label="Lower Safe Bound"),
            *legend_handles,
            Line2D([0], [0], color=UB_COLOR, linewidth=3, label="Upper Safe Bound"),
        ]
    if show_dls:
        for key, _, dls_color, dls_label in config.DLS_LEVELS:
            if any(info["dls_rels"][key] is not None for info in lp_info):
                legend_handles.append(Line2D([0], [0], color=dls_color, linestyle="--", linewidth=3, label=dls_label))

    if is_bdf:
        pass  # barres pleines, une couleur par LP : rien à expliquer au-delà des seuils
    elif show_cba and show_categories:
        cat_handles, cat_labels = _build_category_legend(colors.sort_categories(list(legend_categories)))
        if not show_import_split:
            cat_handles, cat_labels = cat_handles[:-1], cat_labels[:-1]  # retire "Import" : aucun hachurage tracé
        for h, lbl in zip(cat_handles, cat_labels):
            h.set_label(lbl)
        legend_handles += cat_handles
    elif show_cba and show_import_split:
        legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=1, label="Import"))

    if combined:
        # Repère générique (gris neutre) pour la nuance claire du PBA, valable
        # pour tous les LP : les deux nuances distinguent ici CBA/PBA.
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Consumption-based"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Production-based"))
    elif is_bdf:
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Total (France)"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Household categories"))
    else:
        # Une seule comptabilité affichée : les deux nuances distinguent à la
        # place le scénario de référence des autres (voir _solo_color).
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Base year"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Other scenarios"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, len(legend_handles)),
               bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=15)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.975)

    return fig, ax
