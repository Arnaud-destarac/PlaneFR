"""
Figure "roue radiale des limites planétaires" : 1 secteur (wedge) par LP, avec
1 barre radiale par scénario (hauteur = valeur / Lower Bound, donc la Lower
Bound est un cercle de rayon constant = 1L), une bande extérieure annotée
(1 sous-bande par scénario) et un nom de LP + unité par secteur.

Une seule fonction, pilotée par 2 booléens indépendants show_cba/show_pba
(les deux valent True par défaut -- la figure combine alors les deux) :
  - show_cba=True : valeurs consommation. Les données domestique/importé par
    catégorie existent pour le CBA -> chaque barre peut être empilée comme
    create_stacked_bar_chart (dom plein + imp hachuré, par catégorie de
    consommation), juste pliée en radial au lieu de cartésien. show_categories
    et show_import_split (bool, indépendants) activent/désactivent chacune de
    ces deux distinctions ; si les deux sont False, la barre redevient pleine,
    colorée par la couleur du LP.
  - show_pba=True : valeurs production (payload["pba"]). C'est déjà un
    scalaire unique dans les données sources (process_single_subprocess_scenario
    ne le ventile ni par catégorie ni par dom/imp) -> chaque barre reste pleine,
    colorée par une nuance plus claire de la couleur du LP (pour la distinguer
    du CBA quand les deux sont affichés), quels que soient show_categories/
    show_import_split (ignorés côté PBA).
  - show_cba=True ET show_pba=True : les deux comptabilités sur la même
    figure -- pour chaque scénario, 2 barres adjacentes (CBA puis PBA) et 2
    sous-bandes adjacentes dans la bande extérieure.

Titre, légende et sous-titre s'adaptent automatiquement à show_cba/show_pba
(pas besoin de les répéter à chaque appel).

Aucun dégradé de couleur entre scénarios : toutes les barres/segments d'un
même LP (bande extérieure comprise) partagent la couleur de ce LP (couleurs
demandées : vert clair cropland, vert foncé biodiversité, bleu ciel eau bleue,
gris matières premières, marron clair N, marron foncé P, orange GHG). Les
scénarios ne sont distingués que par leur position angulaire (gauche->droite
au sein du secteur, rappelée en sous-titre de la figure) et par le texte de la
bande extérieure.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, config, processing
from .plot_stacked_bar import _build_category_legend

# ============================================================================
# COULEURS
# ============================================================================

LP_COLORS = {
    "N surplus":                 "#DC8968",
    "P surplus":                 "#6D4C41",  
    "Blue water consumption":    "#096DB4",  
    "Cropland use":               "#3B5A17", 
    "GHG emissions":              "#F57C00",
    "Biodiversity loss":          "#B12BB1", 
    "Raw material consumption":   "#757575", 
}
_FALLBACK_LP_COLOR = "#4c78a8"

LB_COLOR = "#0d9a33"    # identique à plot_stacked_bar.py / plot_overshoot.py ("Lower Bound")
UB_COLOR = "#bc270a"    # identique à plot_stacked_bar.py / plot_overshoot.py ("Upper Bound")
DLS_COLOR = "cyan"      # identique à plot_stacked_bar.py ("DLS")

# Aucun dégradé par scénario : toutes les barres/segments d'un LP (bande
# extérieure comprise) partagent la même couleur (celle du LP). Les scénarios
# ne sont distingués que par leur position angulaire au sein du secteur (voir
# le sous-titre "Ordre des scénarios..." sur la figure) et par le texte de la
# bande extérieure.


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
    """Graduations entières (en multiples de la Lower Bound) de 1 à ceil(rel_cap),
    en sous-échantillonnant si trop nombreuses (même idée que
    plot_overshoot._set_break_axis_ticks). 1L est toujours inclus (cercle Lower
    Bound)."""
    r_ceil = max(1, int(np.ceil(rel_cap)))
    ticks = list(range(1, r_ceil + 1))
    if len(ticks) > max_ticks:
        step = max(1, int(np.ceil(r_ceil / max_ticks)))
        ticks = list(range(step, r_ceil + 1, step))
        if ticks[-1] != r_ceil:
            ticks.append(r_ceil)
    return sorted(set(ticks) | {1})


# ============================================================================
# EXTRACTION DES VALEURS, EMPILEMENT PAR CATÉGORIE, ANNOTATION
# ============================================================================


def _extract_value(payload, value_key):
    """CBA = domestique + importé (mêmes totaux que create_synthesis_figure_by_lp) ;
    PBA = payload["pba"] (déjà un scalaire). None si absent/non-fini."""
    if payload is None:
        return None
    if value_key == "pba":
        val = payload.get("pba")
    else:
        val = float(payload["domestique"].sum() + payload["importé"].sum())
    return float(val) if val is not None and np.isfinite(val) else None


def _stack_segments(payload, categories, threshold_lb, group_by_category,
                     show_categories, show_import_split, fallback_color):
    """Décompose l'empreinte CBA d'un (LP, scénario) en segments empilables, en
    unités relatives à la Lower Bound, prêts à dessiner (couleur déjà résolue).
    to_r() doit être appliqué aux BORNES cumulées par l'appelant (pas à la
    hauteur du segment), seule façon de rester correct en échelle non-linéaire
    (radial_power != 1).

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
    if payload is None or threshold_lb is None:
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
        rel_val = abs_val / threshold_lb
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
    title=None,
    figsize=(16, 16),
):
    """Roue radiale : 1 secteur par LP, 1 barre par scénario, hauteur = valeur /
    Lower Bound. Le cercle Lower Bound (vert) est donc à rayon constant = 1L
    sur toute la figure ; l'arc Upper Bound (rouge) est propre à chaque secteur
    (le ratio Upper/Lower Bound varie selon le LP). Bande extérieure : reprend
    l'annotation de create_synthesis_figure_by_lp (valeur absolue pour le
    scénario de référence, variation % pour les autres), repositionnée en
    1 sous-bande par scénario.

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
              scénario. Titre, légende et sous-titre s'adaptent automatiquement.
        show_categories: (CBA uniquement) True = empile 1 segment par catégorie
            de consommation, coloré comme create_stacked_bar_chart. False = pas
            de distinction de catégorie ; la barre prend alors la couleur du LP
            (nuancée par scénario, comme la bande extérieure et les barres PBA).
        show_import_split: (CBA uniquement) True = distingue domestique (plein)
            / importé (hachuré). False = domestique + importé fusionnés dans le
            même segment.
        group_by_category: voir create_stacked_bar_chart -- pertinent seulement
            si show_categories=True et show_import_split=True (ordre
            d'empilement dom/imp entrelacé par catégorie, ou tout dom puis tout
            imp).
        show_dls, dls_df: si show_dls=True, ajoute un arc DLS (plancher social)
            par secteur, quand seuils.xlsx/dls_df fournit une valeur pour ce LP.
        radial_power: exposant de la transformation rayon = (valeur/LB)**radial_power.
            1.0 = échelle linéaire stricte. 0.5 (défaut) = racine carrée. Plus
            la valeur est petite (ex. 1/3), plus l'échelle est "dilatée" : les
            LP en-deçà de leur limite (ex. Blue water consumption, <1L) sont
            étirés vers l'extérieur et donc plus visibles, tandis que les LP en
            fort dépassement (ex. GHG emissions) sont d'autant plus compressés
            -- même principe que sqrt, en plus prononcé. Le cercle Lower Bound
            reste à un rayon constant (=1) quelle que soit la valeur, car 1
            élevé à n'importe quelle puissance vaut toujours 1.
        lp_colors: dict optionnel pour surcharger/étendre LP_COLORS.
        r_max: plafond optionnel de l'axe radial, en multiples de la Lower
            Bound (même unité que les graduations "NL") ; None = auto-calculé
            depuis les données (valeurs + seuils UB/DLS affichés).
        title: préfixe de titre optionnel (ex. "Planetary boundaries assessment
            of French NZE Scenarios") ; le suffixe (" - Consumption-based",
            " - Production-based" ou " - CBA vs PBA") est ajouté automatiquement
            selon show_cba/show_pba. Sans préfixe, seul le suffixe est affiché.

    Returns:
        (fig, ax)
    """
    if not show_cba and not show_pba:
        raise ValueError("show_cba et show_pba ne peuvent pas être tous les deux False.")
    combined = show_cba and show_pba

    reference_idx = config.REFERENCE_SCENARIO_IDX
    first_scenario_data = all_scenarios_data[reference_idx]
    subprocesses = list(first_scenario_data.keys())
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
        threshold_lb = processing.lookup_seuil(seuils_df, config.THRESHOLD_LB_ABS, subprocess_name, require_positive=True)
        threshold_ub = processing.lookup_seuil(seuils_df, config.THRESHOLD_UB_ABS, subprocess_name, require_positive=True)
        unit_text = processing.lookup_first_text(seuils_df, ["Unité d'affichage", "Unité affichage"], subprocess_name)
        floor_ub = None
        if show_dls and dls_df is not None:
            floor_ub = processing.lookup_seuil(dls_df, "Moyenne France", subprocess_name, require_positive=True)

        categories = (first_scenario_data.get(subprocess_name) or {}).get("categories", [])
        legend_categories.update(categories)

        if combined:
            values = []
            for sd in all_scenarios_data:
                payload = sd.get(subprocess_name)
                values.append(_extract_value(payload, "cba"))
                values.append(_extract_value(payload, "pba"))
        else:
            solo_key = "cba" if show_cba else "pba"
            values = [_extract_value(sd.get(subprocess_name), solo_key) for sd in all_scenarios_data]

        if threshold_lb is not None:
            rel_values = [(v / threshold_lb) if v is not None else None for v in values]
            ub_rel = (threshold_ub / threshold_lb) if threshold_ub is not None else None
            dls_rel = (floor_ub / threshold_lb) if floor_ub is not None else None
        else:
            rel_values = [None] * n_slots
            ub_rel = None
            dls_rel = None

        for rel in rel_values + [ub_rel, dls_rel]:
            if rel is not None and np.isfinite(rel):
                max_rel = max(max_rel, rel)

        lp_info.append(dict(
            name=subprocess_name, unit=unit_text, threshold_lb=threshold_lb, categories=categories,
            values=values, rel_values=rel_values, ub_rel=ub_rel, dls_rel=dls_rel,
            color=palette.get(subprocess_name, _FALLBACK_LP_COLOR),
        ))

    # Les graduations "NL" sont calculées AVANT le plafond radial : la dernière
    # (rel_ticks[-1], toujours = ceil(rel_cap_input) d'après _nice_radial_ticks)
    # devient elle-même ce plafond, pour que la bande extérieure démarre pile
    # sur ce dernier cercle -- pas d'espace mort entre la plus haute barre et
    # la bande (l'arrondi au L entier supérieur joue déjà le rôle de marge).
    rel_cap_input = r_max if r_max is not None else max_rel
    rel_ticks = _nice_radial_ticks(rel_cap_input)
    rel_cap = float(rel_ticks[-1])
    r_max_plot = to_r(rel_cap)
    ring_thickness = r_max_plot * 0.11
    r_ring_start = r_max_plot
    r_ring_end = r_ring_start + ring_thickness
    r_label = r_max_plot * 0.78

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    # Marges par défaut de matplotlib très généreuses pour un unique subplot
    # (~12% de chaque côté) : on les resserre pour que le cercle remplisse la
    # figure, avec juste la place nécessaire au-dessus pour le titre/sous-titre
    # et en-dessous pour la légende (positionnés juste après, au ras de ces
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
            continue  # tracé séparément, en gras, comme cercle Lower Bound
        ax.plot(theta_full, np.full_like(theta_full, to_r(t)), color="#e0e0e0", linewidth=0.7, zorder=1)

    for b in geo["boundaries"]:
        ax.plot([b, b], [0, r_ring_end], color="#bdbdbd", linewidth=0.8, zorder=1)
        offset = geo["wedge_span"] * 0.02
        for t in rel_ticks:
            ax.text(b + offset, to_r(t), f"{t}L", ha="left", va="center", fontsize=12,
                     color="#555555", rotation=_radial_rotation_deg(b), rotation_mode="anchor",
                     bbox=dict(facecolor="white", edgecolor="white", alpha=0.7, pad=0.5), zorder=3)

    # DEUXIÈME PASSE : barres, bande extérieure, arcs UB/DLS, nom de LP
    for lp_idx, info in enumerate(lp_info):
        centers = geo["centers"][lp_idx]
        bar_width = geo["bar_width"]
        ring_centers = geo["ring_centers"][lp_idx]

        def _draw_cba_bar(theta, scenario_idx):
            payload = all_scenarios_data[scenario_idx].get(info["name"])
            segments = _stack_segments(payload, info["categories"], info["threshold_lb"], group_by_category,
                                        show_categories, show_import_split, info["color"])
            for rel_bottom, rel_top, color, is_import in segments:
                ax.bar([theta], [to_r(rel_top) - to_r(rel_bottom)], width=bar_width, bottom=to_r(rel_bottom),
                       color=color, edgecolor="white", linewidth=0.4,
                       hatch="/" if is_import else None, zorder=5)

        # PBA : toujours en nuance plus claire de la couleur du LP (au lieu
        # d'un hachurage) -- que ce soit la seule comptabilité affichée ou en
        # complément du CBA en mode combiné.
        pba_color = colors.get_light_shade(info["color"], 1.5)

        if combined:
            # 2 barres adjacentes par scénario -- CBA (empilée comme
            # _draw_cba_bar) puis PBA (toujours pleine, nuance claire).
            for scenario_idx, theta in enumerate(centers[0::2]):
                _draw_cba_bar(theta, scenario_idx)
            pba_centers = centers[1::2]
            pba_heights = [to_r(rel) if rel is not None else 0.0 for rel in info["rel_values"][1::2]]
            ax.bar(pba_centers, pba_heights, width=bar_width, bottom=0.0,
                   color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        elif show_cba:
            # Empreinte détaillée disponible (domestique/importé par catégorie) :
            # barre empilée selon show_categories/show_import_split (les deux à
            # False -> 1 segment plein, coloré par le LP). to_r() est appliqué
            # aux BORNES cumulées (pas à la hauteur du segment), seule façon
            # correcte de rester cohérent avec le cercle LB/l'arc UB en échelle
            # non-linéaire (radial_power != 1).
            for scenario_idx, theta in enumerate(centers):
                _draw_cba_bar(theta, scenario_idx)
        else:
            # PBA seul : une seule valeur scalaire par (LP, scénario) dans les
            # données sources -> pas de ventilation possible, barre pleine.
            heights = [to_r(rel) if rel is not None else 0.0 for rel in info["rel_values"]]
            ax.bar(centers, heights, width=bar_width, bottom=0.0,
                   color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)

        # Bande extérieure : pavage complet du secteur (ring_centers/ring_width,
        # sans le gap_frac des barres) pour ne laisser aucun vide entre
        # scénarios ni entre secteurs. En mode combiné, la moitié PBA reprend
        # la nuance claire de la barre correspondante ; sinon toute la bande
        # suit la couleur (pleine ou claire) de la comptabilité affichée.
        if combined:
            ax.bar(ring_centers[0::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=info["color"], edgecolor="white", linewidth=0.6, zorder=5)
            ax.bar(ring_centers[1::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        else:
            ring_color = info["color"] if show_cba else pba_color
            ax.bar(ring_centers, [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=ring_color, edgecolor="white", linewidth=0.6, zorder=5)

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
            is_pba_by_slot = [show_pba] * n_slots

        r_ring_mid = (r_ring_start + r_ring_end) / 2
        for i, theta in enumerate(ring_centers):
            val = info["values"][i]
            if val is None:
                continue
            text, color = _annotate(val, ref_by_slot[i], is_ref_by_slot[i], is_pba_by_slot[i])
            ax.text(theta, r_ring_mid, text, ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color, zorder=9)

        if info["ub_rel"] is not None:
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, to_r(info["ub_rel"])), color=UB_COLOR,
                    linewidth=3.0, solid_capstyle="butt", zorder=8)

        if info["dls_rel"] is not None:
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, to_r(info["dls_rel"])), color=DLS_COLOR,
                    linestyle="--", linewidth=3.0, zorder=8)

        bisector = geo["boundaries"][lp_idx] + geo["wedge_span"] / 2
        label = f"{info['name']}\n({info['unit']})" if info["unit"] else info["name"]
        txt = ax.text(bisector, r_label, label, ha="center", va="center",
                       fontsize=14, fontweight="bold", color=info["color"], linespacing=1.4,
                       rotation=_tangential_rotation_deg(bisector), rotation_mode="anchor", zorder=9)
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    # Lower Bound : cercle plein, rayon constant = 1L (point fixe de to_r, quel
    # que soit radial_power), tracé par-dessus les barres.
    ax.plot(theta_full, np.full_like(theta_full, to_r(1.0)), color=LB_COLOR, linewidth=3.2, zorder=8)

    ax.set_rlim(0, r_ring_end * 1.02)

    # Légende : lignes de seuil communes aux deux figures, puis les catégories
    # si affichées (CBA avec show_categories=True -- explique l'habillage des
    # barres empilées) et/ou l'entrée "Import" seule si la distinction dom/imp
    # est affichée sans les catégories. Aucune entrée par scénario : les
    # barres/segments ne sont plus nuancées par scénario (voir plus haut).
    legend_handles = [
        Line2D([0], [0], color=LB_COLOR, linewidth=3, label="Lower Bound"),
        Line2D([0], [0], color=UB_COLOR, linewidth=3, label="Upper Bound"),
    ]
    if show_dls:
        legend_handles.append(Line2D([0], [0], color=DLS_COLOR, linestyle="--", linewidth=3, label="DLS"))

    if show_cba and show_categories:
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
        # pour tous les LP.
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Consumption-based"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Production-based"))

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, len(legend_handles)),
               bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=15)

    # Ordre (fixe) des scénarios au sein de chaque secteur, en légende textuelle
    # puisque les barres CBA n'encodent plus le scénario par couleur (seule la
    # bande extérieure et la position angulaire l'identifient). Positionné juste
    # au-dessus de la nouvelle borne haute des axes (subplots_adjust top=0.91).
    caption = "Scenario order : " + " → ".join(scenario_names)
    fig.text(0.5, 0.94, caption, ha="center", fontsize=16, style="italic", color="#333333")
    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.975)

    return fig, ax
