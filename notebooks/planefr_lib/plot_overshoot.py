"""
Figure "Overshoot / Safe Operating Space" : 1 ligne par limite planétaire (LP),
position de chaque scénario en abscisse relative à la limite basse (1L = seuil
d'égalité), fond dégradé vert -> orange -> violet (zone sûre -> risque
croissant -> zone à haut risque), rupture d'axe si un scénario dépasse
x_main_max fois la limite basse.

Deux fonctions publiques, qui produisent une figure du même type mais sont
volontairement INDÉPENDANTES l'une de l'autre (aucun rendu ni helper partagé
au-delà des primitives de dessin génériques ci-dessous, jamais modifiées
depuis leur création) : modifier l'une ne doit jamais changer le comportement
de l'autre.

  - create_overshoot_safe_space_figure : implémentation d'origine, budgets lus
    dans seuils.xlsx (statiquement, ou recalculés par partage égal per capita
    via sharing_principle). Sert les deux usages historiques du projet :
      * Overshoot multi-scénarios : valeurs absolues, 1 bulle par scénario (CBA
        uniquement, pas de comptabilité PBA calculée pour ce cas) ;
      * Figure comparaison : valeurs par habitant, 2 bulles par scénario (CBA
        pleine + PBA hachurée).
  - create_overshoot_safe_space_figure_by_region : valeurs absolues (rien n'est
    divisé par la population, ni les empreintes ni les budgets), budget de chaque
    pays/région obtenu en multipliant le budget mondial par sa part lue dans
    budget_shares.xlsx (principes "EPC"/"CTR"). Implémentation entièrement
    autonome (ses propres versions privées de tri des LP, dessin des bulles,
    dimensionnement des lignes...), qui n'appelle jamais le code de
    create_overshoot_safe_space_figure.

Par décision explicite : Figure comparaison.ipynb est la référence pour les
valeurs par défaut (couleur de limite haute, absence d'étiquette de valeur sur
les bulles, scenario_style_map) — voir planefr_lib.colors.scenario_style_map
et le README du refactor pour le détail des divergences résolues.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import colors, config, processing

# Taille de police des graduations d'axe X en "nL" (ex: 1L, 2L... 7L, 8L...),
# partagée par l'axe principal et l'axe de rupture pour qu'ils restent identiques.
AXIS_TICK_LABEL_FONTSIZE = 13

# title_above_bar=True uniquement — hauteur (fraction de hauteur de barre)
# occupée par les 2 lignes du titre (nom + unité) + l'espace entre elles, à
# fontsize=13. Sert à dimensionner automatiquement `hspace` (l'espace entre
# barres) à partir des deux écarts réglables title_gap_below_bar/
# title_gap_above_bar, pour que le titre ait toujours la place de tenir.
TITLE_BLOCK_HEIGHT = 0.35

# ============================================================================
# HELPERS GÉNÉRIQUES, PARTAGÉS PAR LES DEUX FIGURES (jamais modifiés depuis
# leur création : pures primitives de dessin/formatage, sans dépendance à la
# façon dont les budgets sont obtenus -- aucun risque qu'une évolution propre
# à l'une des deux figures ne déborde sur l'autre).
# ============================================================================


def _fmt_abs(value):
    """Formate une valeur absolue pour annotation (séparateur de milliers insécable)."""
    if value is None or not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _total_footprint(lp_payload):
    """Empreinte totale (domestique + importée) pour un sous-processus."""
    return float(lp_payload["domestique"].sum() + lp_payload["importé"].sum())


def _get_lp_unit(seuils_df, subprocess_name, unit_row):
    """Récupère l'unité d'affichage pour un LP, ou "" si absente."""
    return processing.lookup_first_text(seuils_df, unit_row, subprocess_name)


def _draw_gradient_segment(ax, x0, x1, seg_colors, y0=0.10, y1=0.90):
    """Dessine un segment horizontal avec dégradé continu entre `seg_colors`."""
    if x1 <= x0:
        return
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("lp_grad", seg_colors)
    ax.imshow(grad, extent=[x0, x1, y0, y1], aspect="auto", cmap=cmap, interpolation="bicubic", zorder=0)


def _draw_lp_background(ax, x_min, x_max):
    """Fond safe (vert) -> risque croissant (jaune/orange) -> haut risque (rouge/violet),
    avec transitions à SAFE_LIMIT_REL (1L) et RISK_TRANSITION_REL (2L)."""
    _draw_gradient_segment(ax, max(x_min, 0), min(x_max, config.SAFE_LIMIT_REL), ["#0b7d3e", "#73bf44"])
    _draw_gradient_segment(ax, max(x_min, config.SAFE_LIMIT_REL), min(x_max, config.RISK_TRANSITION_REL),
                            ["#f5dd3d", "#eb8f1e"])
    _draw_gradient_segment(ax, max(x_min, config.RISK_TRANSITION_REL), x_max,
                            ["#e7731c", "#d62324", "#8f1f64", "#4f256d"])


def _draw_above_break_uniform(ax, x_min, x_max):
    """Au-delà de x_main_max, couleur uniforme égale à la couleur atteinte en bout de dégradé."""
    if x_max <= x_min:
        return
    ax.axvspan(x_min, x_max, ymin=0.10, ymax=0.90, facecolor="#4f256d", edgecolor="none", zorder=0)


def _set_break_axis_ticks(ax_ext):
    """Graduations entières lisibles sur le segment de droite après la rupture d'axe."""
    x_min, x_max = ax_ext.get_xlim()
    int_min, int_max = int(np.floor(x_min)), int(np.ceil(x_max))
    if int_max < int_min:
        int_min, int_max = int_max, int_min

    ticks = list(range(int_min, int_max + 1))
    if len(ticks) > 8:
        step = max(1, int(np.ceil((int_max - int_min) / 7)))
        ticks = list(range(int_min, int_max + 1, step))
        if ticks[-1] != int_max:
            ticks.append(int_max)

    ax_ext.set_xticks(ticks)
    ax_ext.set_xticklabels([f"{t}L" for t in ticks], fontsize=AXIS_TICK_LABEL_FONTSIZE, fontweight="bold")


def _plot_sensitivity_segment(ax, x_min, x_max, y_pos, color, cap_half_height=0.04,
                               linewidth=2.2, zorder=1.5):
    """Dessine l'intervalle de confiance (sensibilité au principe de partage,
    sharing_principle="all") d'un point : segment horizontal entre `x_min` et
    `x_max`, avec un petit trait vertical à chaque extrémité (style barre
    d'erreur). Dessiné sous la bulle (zorder < 2), qui masque donc le centre du
    segment et n'en laisse dépasser que les extrémités."""
    if x_max is None or x_min is None or x_max <= x_min:
        return
    ax.plot([x_min, x_max], [y_pos, y_pos], color=color, linewidth=linewidth,
            solid_capstyle="butt", zorder=zorder)
    for x in (x_min, x_max):
        ax.plot([x, x], [y_pos - cap_half_height, y_pos + cap_half_height], color=color,
                linewidth=linewidth, zorder=zorder)


def _draw_lp_title_above(ax, name, unit_text, ceiling, gap_below_bar, gap_above_bar, fontsize=13):
    """Affiche le nom du LP au-dessus de la barre, aligné à gauche sur le début
    de l'axe (x=0), au lieu du libellé habituel dans la marge de gauche.
    L'unité, si présente, est systématiquement affichée sur une seconde ligne
    en-dessous du nom.

    Les deux lignes sont ancrées indépendamment, pour que chaque écart soit
    réglable sans affecter l'autre :
      - l'unité (ou le nom, s'il n'y a pas d'unité) est ancrée par le bas
        (va="bottom") à `gap_below_bar` au-dessus de la barre qu'elle nomme
        (y=1.0 = haut de cette barre) — "collée" à la barre par défaut ;
      - le nom est ancré par le haut (va="top") à `gap_above_bar` en-dessous de
        `ceiling`, la limite haute disponible pour cette ligne (typiquement le
        bas de la barre du dessus — cf. appelant).
    """
    if unit_text:
        ax.text(0.0, 1.0 + gap_below_bar, f"({unit_text})", transform=ax.transAxes, ha="left", va="bottom",
                fontsize=fontsize, color="#4a4a4a", clip_on=False)
        ax.text(0.0, ceiling - gap_above_bar, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=fontsize, fontweight="bold", clip_on=False)
    else:
        ax.text(0.0, 1.0 + gap_below_bar, name, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=fontsize, fontweight="bold", clip_on=False)


# ============================================================================
# FIGURE PRINCIPALE : create_overshoot_safe_space_figure
# ============================================================================
# Implémentation d'origine, self-contained (elle n'appelle aucun code partagé
# avec create_overshoot_safe_space_figure_by_region au-delà des primitives de
# dessin génériques ci-dessus) -- ne pas la modifier au passage d'un changement
# destiné à create_overshoot_safe_space_figure_by_region.


def _find_france_scenario_idx(scenario_names):
    """Index du scénario "France" dans scenario_names, utilisé pour trier les LP
    par dépassement CBA (voir _sort_subprocesses_by_france_overshoot).

    Cherche un nom de scénario contenant "france" (insensible à la casse, ex.
    "2019_France" dans la figure comparaison). Si aucun ne correspond (cas
    multi-scénarios où tous les scénarios sont déjà la France, ex. Base_year_2015/
    TREND_2050/...), retombe sur le scénario de référence
    (config.REFERENCE_SCENARIO_IDX)."""
    for idx, name in enumerate(scenario_names):
        if "france" in name.lower():
            return idx
    return config.REFERENCE_SCENARIO_IDX


def _sort_subprocesses_by_france_overshoot(
    subprocesses, scenario_names, all_scenarios_data, seuils_df,
    threshold_lb_row, pop_df, sharing_principle,
):
    """Trie les LP (lignes de la figure) par dépassement croissant du scénario
    France (max entre CBA et PBA, si PBA présent) : plus la bulle France la plus
    à droite sur une barre est loin, plus cette barre est affichée bas par
    rapport aux autres."""
    france_data = all_scenarios_data[_find_france_scenario_idx(scenario_names)]

    def sort_key(subprocess_name):
        payload = france_data.get(subprocess_name)
        lb_val = processing.lookup_threshold(seuils_df, threshold_lb_row, subprocess_name,
                                              pop_df=pop_df, sharing_principle=sharing_principle)
        if payload is None or not lb_val:
            return float("-inf")
        rel_cba = _total_footprint(payload) / lb_val
        abs_pba = payload.get("pba")
        rel_pba = abs_pba / lb_val if abs_pba is not None else float("-inf")
        rel_max = max(rel_cba, rel_pba)
        return rel_max if np.isfinite(rel_max) else float("-inf")

    return sorted(subprocesses, key=sort_key)


def _plot_bubble(ax, rel_x, y_pos, style, hatched=False, hatch_color=None, value_label=None):
    """Dessine une bulle de scénario sur `ax`.

    hatched=True : bulle hachurée (PBA-accounting) ; hatch_color en contrôle la
    couleur (noir par défaut, blanc pour les scénarios "trend", pour rester
    lisible sur un fond sombre).
    value_label, si fourni, affiche le texte à droite de la bulle (avec un
    liseré sombre pour rester lisible sur le dégradé) — utilisé uniquement
    quand show_value_labels=True dans create_overshoot_safe_space_figure.
    """
    sc = ax.scatter(rel_x, y_pos, s=300, marker="o", facecolor=style["face"],
                     edgecolor=style["edge"], linewidth=1.5, zorder=2)
    if hatched:
        sc.set_hatch("///")
        if hatch_color is not None:
            sc.set_edgecolor(hatch_color)

    if value_label is not None:
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        txt = ax.text(rel_x + x_span * 0.015, y_pos, value_label, color=style["face"],
                       fontsize=8.2, ha="left", va="center", zorder=11, clip_on=True)
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#2b2b2b")])


def create_overshoot_safe_space_figure(
    all_scenarios_data, seuils_df, scenario_names,
    threshold_lb_row, threshold_ub_row, unit_row,
    ub_color="#bc270a", x_main_max=8.0, display_bounds=True,
    show_pba=False, show_value_labels=True, show_value_bounds=True,
    title_above_bar=False,
    title_gap_below_bar=0.01, title_gap_above_bar=0.01,
    title="Overshoot relative to the Safe Operating Space",
    sharing_principle="Equality 2100", pop_df=None,
):
    """Figure multi-LP de type Overshoot / Safe Operating Space.

    Axe X relatif à la limite basse (LB = 1L) : limite haute = UB/LB, empreinte
    scénario = Empreinte/LB.

    Args:
        all_scenarios_data: liste de {sous_processus: payload}, un élément par
            scénario, dans l'ordre de scenario_names. payload doit contenir
            "domestique"/"importé" (utilisés par _total_footprint pour le CBA),
            et optionnellement "pba" (float) si show_pba=True.
        threshold_lb_row, threshold_ub_row, unit_row: noms des lignes à lire
            dans seuils_df — passer les constantes _ABS de planefr_lib.config
            pour la figure en valeurs absolues, _PER_CAPITA pour la figure par
            habitant. Le 3e seuil ("Lower Safe Bound", vert clair) est déduit
            automatiquement de threshold_lb_row via config.THRESHOLD_LOWER_FOR
            (pas de paramètre dédié). Les 3 traits reprennent le nom des lignes
            "brutes" de seuils.xlsx/feuille "Synthèse" : "Lower Safe Bound"
            (vert clair), "Safe Limit" (vert, seuil historique threshold_lb_row),
            "Upper Safe Bound" (rouge, threshold_ub_row).
        display_bounds: True (défaut) affiche les 3 traits (Lower Safe Bound,
            Safe Limit, Upper Safe Bound). False n'affiche que Safe Limit
            (masque Lower Safe Bound et Upper Safe Bound, ainsi que leurs
            entrées de légende).
        show_pba: si True, cherche une clé "pba" dans chaque payload et trace
            une seconde bulle hachurée (production-based accounting).
        show_value_labels: si True, affiche la valeur absolue à côté de chaque
            bulle (uniquement utilisé par la figure en valeurs absolues).
        show_value_bounds: si True (défaut), affiche la valeur absolue
            au-dessus du trait Safe Limit ainsi que l'unité du LP (à côté du
            nom, en marge de gauche ou au-dessus de la barre selon
            title_above_bar). Si False, masque ces deux éléments. Les valeurs
            de Lower Safe Bound et Upper Safe Bound ne sont, elles, jamais
            affichées. Sans effet sur les traits eux-mêmes (voir
            display_bounds).
        title_above_bar: si False (défaut), nom du LP + unité affichés dans la
            marge de gauche, comme avant. Si True, affichés au-dessus de la
            barre, alignés à gauche sur son début (x=0), unité systématiquement
            sur une seconde ligne sous le nom. La marge de gauche de la figure
            est alors réduite au minimum puisque les titres n'y occupent plus
            de place (le début des barres redevient l'élément le plus à gauche).
        title_gap_below_bar: (title_above_bar=True uniquement) écart, en
            fraction de hauteur de barre, entre le titre et la barre qu'il
            nomme (en-dessous) — plus la valeur est petite, plus le titre est
            "collé" à sa barre.
        title_gap_above_bar: (title_above_bar=True uniquement) écart, en
            fraction de hauteur de barre, entre le titre et la barre du dessus
            (ou, pour la 1ère barre, les graduations de l'axe X partagé).
        sharing_principle: si fourni ("Equality 2050"/"Equality 2100"/"Equality
            2019"), LB/UB sont recalculés à la volée par un partage égal per
            capita du budget mondial pour la période de référence associée
            (voir config.SHARING_PRINCIPLE_POPULATION_ROW et
            processing.compute_sharing_seuil), au lieu d'être lus statiquement
            dans seuils_df via threshold_lb_row/threshold_ub_row. Nécessite
            pop_df. Si None (défaut), comportement inchangé.
            Si "all" : fond, traits de seuil et position des bulles restent
            ceux de "Equality 2100" (comportement identique à
            sharing_principle="Equality 2100"), mais chaque bulle reçoit en
            plus un segment horizontal (intervalle de confiance, avec petits
            traits verticaux aux extrémités) représentant sa sensibilité aux
            3 principes de partage possibles : la borne min/max du segment est
            le min/max de l'overshoot du point recalculé sous chacun des 3
            principes (dont "Equality 2100" lui-même). Le segment est blanc
            pour les bulles CBA (pleines) et noir pour les bulles PBA
            (hachurées, show_pba=True), et apparaît dans la légende sous
            "sensitivity to allocation principles".
        pop_df: feuille "Population" de seuils.xlsx (io.load_population_df) --
            requis seulement si sharing_principle est fourni.

    Returns:
        matplotlib.figure.Figure
    """
    if not all_scenarios_data:
        raise ValueError("all_scenarios_data est vide")

    # sharing_principle="all" : fond/traits/bulles utilisent "Equality 2100"
    # (voir docstring) ; la sensibilité aux 3 principes est calculée à part,
    # par point, plus bas dans la boucle.
    show_sensitivity = sharing_principle == "all"
    display_sharing_principle = "Equality 2100" if show_sensitivity else sharing_principle

    first_scenario_data = all_scenarios_data[config.REFERENCE_SCENARIO_IDX]
    subprocesses = list(first_scenario_data.keys())
    subprocesses = _sort_subprocesses_by_france_overshoot(
        subprocesses, scenario_names, all_scenarios_data, seuils_df,
        threshold_lb_row, pop_df, display_sharing_principle,
    )
    n_lp = len(subprocesses)

    scenario_style = colors.scenario_style_map(scenario_names)
    y_levels = np.linspace(0.82, 0.18, len(scenario_names)) if len(scenario_names) > 1 else np.array([0.5])
    scenario_to_y = dict(zip(scenario_names, y_levels))

    row_height_factor = 1.35 if title_above_bar else 1.2
    # hspace dimensionné pour que le bloc de titre (nom + unité) tienne
    # toujours entre les deux écarts demandés, quels que soient leurs réglages.
    hspace = (title_gap_below_bar + title_gap_above_bar + TITLE_BLOCK_HEIGHT) if title_above_bar else 0.12
    fig = plt.figure(figsize=(18, max(8, row_height_factor * n_lp + 1.8)))
    gs = fig.add_gridspec(n_lp, 2, width_ratios=[7.0, 3.0], hspace=hspace, wspace=0.02)

    top_main_ax = None

    for row_idx, subprocess_name in enumerate(subprocesses):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax_ext = fig.add_subplot(gs[row_idx, 1], sharey=ax)
        ax.set_zorder(3)
        ax_ext.set_zorder(2)
        ax_ext.patch.set_alpha(0.0)
        if row_idx == 0:
            top_main_ax = ax

        unit_text = _get_lp_unit(seuils_df, subprocess_name, unit_row) if show_value_bounds else ""
        lb_val = processing.lookup_threshold(seuils_df, threshold_lb_row, subprocess_name,
                                              pop_df=pop_df, sharing_principle=display_sharing_principle)
        ub_val = processing.lookup_threshold(seuils_df, threshold_ub_row, subprocess_name,
                                              pop_df=pop_df, sharing_principle=display_sharing_principle)
        threshold_lower_row = config.THRESHOLD_LOWER_FOR.get(threshold_lb_row)
        lower_val = (
            processing.lookup_threshold(seuils_df, threshold_lower_row, subprocess_name,
                                         pop_df=pop_df, sharing_principle=display_sharing_principle)
            if threshold_lower_row else None
        )

        # sharing_principle="all" : LB recalculé sous chacun des 3 principes de
        # partage pour ce LP, pour dériver la sensibilité de chaque point
        # (indépendant du scénario, donc calculé une seule fois par ligne).
        sensitivity_lb_vals = {}
        if show_sensitivity:
            for sp in config.SHARING_PRINCIPLE_POPULATION_ROW:
                sp_val = processing.lookup_threshold(seuils_df, threshold_lb_row, subprocess_name,
                                                      pop_df=pop_df, sharing_principle=sp)
                if sp_val:
                    sensitivity_lb_vals[sp] = sp_val

        if lb_val is None:
            ax.text(0.5, 0.5, f"{subprocess_name}: limite basse manquante", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#7a7a7a")
            ax.set_xlim(0, x_main_max)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax_ext.set_visible(False)
            continue

        ub_rel = (ub_val / lb_val) if ub_val is not None else None
        lower_rel = (lower_val / lb_val) if lower_val is not None else None

        # Calcul des positions relatives (CBA, et PBA si demandé) par scénario
        points = []
        for scenario_idx, scenario_data in enumerate(all_scenarios_data):
            if subprocess_name not in scenario_data:
                continue
            payload = scenario_data[subprocess_name]
            abs_cba = _total_footprint(payload)
            rel_cba = abs_cba / lb_val
            if not np.isfinite(rel_cba):
                continue

            abs_pba = payload.get("pba") if show_pba else None
            rel_pba = abs_pba / lb_val if (abs_pba is not None and np.isfinite(abs_pba)) else None

            rel_cba_min = rel_cba_max = None
            rel_pba_min = rel_pba_max = None
            if show_sensitivity and sensitivity_lb_vals:
                cba_variants = [abs_cba / v for v in sensitivity_lb_vals.values()]
                rel_cba_min, rel_cba_max = min(cba_variants), max(cba_variants)
                if abs_pba is not None and np.isfinite(abs_pba):
                    pba_variants = [abs_pba / v for v in sensitivity_lb_vals.values()]
                    rel_pba_min, rel_pba_max = min(pba_variants), max(pba_variants)

            sname = scenario_names[scenario_idx]
            points.append({
                "sname": sname, "y_pos": scenario_to_y[sname],
                "abs_cba": abs_cba, "rel_cba": rel_cba,
                "abs_pba": abs_pba, "rel_pba": rel_pba,
                "rel_cba_min": rel_cba_min, "rel_cba_max": rel_cba_max,
                "rel_pba_min": rel_pba_min, "rel_pba_max": rel_pba_max,
            })

        # Dépassement d'axe : agrège TOUTES les valeurs x présentes sur cette
        # ligne (pas juste cba/pba en dur), pour rester correct si une autre
        # série s'ajoute un jour.
        all_x = [p["rel_cba"] for p in points] + [p["rel_pba"] for p in points if p["rel_pba"] is not None]
        overflow = [x for x in all_x if x > x_main_max]
        use_break = len(overflow) > 0

        ax.set_xlim(0, x_main_max)
        ax.set_ylim(0, 1)
        _draw_lp_background(ax, 0, x_main_max)

        if use_break:
            x2_min = min(overflow) * 0.995
            x2_max = max(overflow) * 1.02
            if (x2_max - x2_min) < 0.5:
                mid = 0.5 * (x2_min + x2_max)
                x2_min, x2_max = mid - 0.25, mid + 0.25

            ax_ext.set_xlim(x2_min, x2_max)
            ax_ext.set_ylim(0, 1)
            _draw_above_break_uniform(ax_ext, x2_min, x2_max)
            _set_break_axis_ticks(ax_ext)

            d = 0.012
            kw_main = dict(transform=ax.transAxes, color="#3a3a3a", clip_on=False, linewidth=1.2)
            kw_ext = dict(transform=ax_ext.transAxes, color="#3a3a3a", clip_on=False, linewidth=1.2)
            ax.plot((1 - d, 1 + d), (-d, +d), **kw_main)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_main)
            ax_ext.plot((-d, +d), (-d, +d), **kw_ext)
            ax_ext.plot((-d, +d), (1 - d, 1 + d), **kw_ext)
        else:
            ax_ext.set_visible(False)

        # Traits + valeurs des limites (indépendants du dégradé de fond) : Safe
        # Limit (vert, seuil historique) toujours affiché, valeur visible
        # uniquement si show_value_bounds=True ; Lower/Upper Safe Bound (vert
        # clair/rouge) uniquement si display_bounds=True, jamais leur valeur.
        if display_bounds and lower_rel is not None and lower_rel >= 0:
            ax.axvline(lower_rel, ymin=0.10, ymax=0.90, color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3, zorder=3)
        ax.axvline(1, ymin=0.10, ymax=0.90, color="#0d9a33", linewidth=3, zorder=3)
        if show_value_bounds:
            ax.text(1, 0.93, _fmt_abs(lb_val), color="#0d9a33", fontsize=11, fontweight="bold",
                    ha="center", va="bottom", zorder=5)
        if display_bounds and ub_rel is not None and ub_rel <= x_main_max:
            ax.axvline(ub_rel, ymin=0.10, ymax=0.90, color=ub_color, linewidth=3, zorder=3)

        # Bulles CBA (pleines) + PBA (hachurées, si show_pba), avec segment de
        # sensibilité (sharing_principle="all") dessiné sous chaque bulle.
        for p in points:
            style = scenario_style.get(p["sname"], {"face": "#b7c4d3", "edge": "#44505c"})

            in_ext_cba = use_break and p["rel_cba"] > x_main_max and ax_ext.get_visible()
            target_ax_cba = ax_ext if in_ext_cba else ax
            if p["rel_cba_min"] is not None:
                _plot_sensitivity_segment(target_ax_cba, p["rel_cba_min"], p["rel_cba_max"], p["y_pos"],
                                           color="white")
            _plot_bubble(target_ax_cba, p["rel_cba"], p["y_pos"], style, hatched=False,
                         value_label=_fmt_abs(p["abs_cba"]) if show_value_labels else None)

            if p["rel_pba"] is not None:
                hatch_color = "white" if "trend" in p["sname"].lower() else "black"
                in_ext_pba = use_break and p["rel_pba"] > x_main_max and ax_ext.get_visible()
                target_ax_pba = ax_ext if in_ext_pba else ax
                if p["rel_pba_min"] is not None:
                    _plot_sensitivity_segment(target_ax_pba, p["rel_pba_min"], p["rel_pba_max"], p["y_pos"],
                                               color="black")
                _plot_bubble(target_ax_pba, p["rel_pba"], p["y_pos"], style, hatched=True,
                             hatch_color=hatch_color,
                             value_label=_fmt_abs(p["abs_pba"]) if show_value_labels else None)

        # Nom du LP + unité : dans la marge de gauche (défaut), ou au-dessus
        # de la barre (title_above_bar=True). `ceiling` (= 1.0 + hspace, le bas
        # de la barre du dessus) est la même formule pour toutes les lignes, y
        # compris la 1ère : hspace est dimensionné (cf. plus haut) pour laisser
        # aussi la place aux graduations de l'axe X partagé au-dessus d'elle.
        if title_above_bar:
            _draw_lp_title_above(ax, subprocess_name, unit_text, 1.0 + hspace,
                                  title_gap_below_bar, title_gap_above_bar)
        else:
            ax.text(-0.03, 0.57, subprocess_name, transform=ax.transAxes, ha="right", va="center",
                    fontsize=13, fontweight="bold")
            if unit_text:
                ax.text(-0.03, 0.39, f"({unit_text})", transform=ax.transAxes, ha="right", va="center",
                        fontsize=13, color="#4a4a4a")

        # Habillage épuré
        for axis in (ax, ax_ext):
            if not axis.get_visible():
                continue
            axis.set_yticks([])
            axis.grid(False)
            for spine in ("left", "right", "bottom"):
                axis.spines[spine].set_visible(False)
            axis.spines["top"].set_visible((axis is ax and row_idx == 0) or (axis is ax_ext and use_break))
            axis.xaxis.tick_top()
            axis.tick_params(axis="x", length=4, pad=4)

        if row_idx != 0:
            ax.tick_params(axis="x", labeltop=False, top=False)
        if use_break and ax_ext.get_visible():
            ax_ext.tick_params(axis="x", labeltop=True, top=True, labelsize=AXIS_TICK_LABEL_FONTSIZE, length=3, pad=2)

    # Axe X principal commun (1L, 2L, 3L...) + labels de zone, sur la 1ère ligne
    if top_main_ax is not None:
        top_main_ax.set_xticks(np.arange(1, int(x_main_max) + 1))
        top_main_ax.set_xticklabels([f"{i}L" for i in range(1, int(x_main_max) + 1)], fontsize=AXIS_TICK_LABEL_FONTSIZE, fontweight="bold")
        # Position de la spine du haut en fraction d'axes (et non en points via
        # "outward") : un décalage en points n'est pas fiable ici, car
        # plt.tight_layout() (utilisé plus bas) recalcule la hauteur réelle des
        # axes sans être compatible avec ce déplacement de spine (cf. warning
        # matplotlib), ce qui pouvait faire atterrir la spine/les graduations
        # en plein milieu du titre de la 1ère barre (ex. "Blue water
        # consumption") au lieu d'au-dessus. En fraction d'axes, la position
        # reste toujours au-dessus du bloc de titre (ceiling), quelle que soit
        # la taille finale de la figure.
        top_spine_y = (1.0 + hspace + 0.03) if title_above_bar else 1.03
        top_main_ax.spines["top"].set_position(("axes", top_spine_y))
        top_main_ax.tick_params(axis="x", pad=8, length=5, labelsize=AXIS_TICK_LABEL_FONTSIZE)
        zone_label_y = (top_spine_y + 0.45) if title_above_bar else 1.44
        for txt, color, x_pos in [
            ("Safe Operating Space", "#0b7d3e", 0.00),
            ("Increasing Risk", "#d47818", 0.20),
            ("High-Risk Zone", "#7a1f54", 0.40),
        ]:
            top_main_ax.text(x_pos, zone_label_y, txt, transform=top_main_ax.transAxes, color=color,
                              fontsize=11, fontweight="bold", ha="left")

    # Légende : scénarios + limites, et type de comptabilité si show_pba
    scenario_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=scenario_style.get(n, {"face": "#b7c4d3"})["face"],
               markeredgecolor=scenario_style.get(n, {"edge": "#44505c"})["edge"],
               markeredgewidth=1.3, markersize=15, label=n)
        for n in scenario_names
    ]
    limit_handles = [Line2D([0], [0], color="#0d9a33", linewidth=3.5, label="Safe Limit")]
    if display_bounds:
        limit_handles = [
            Line2D([0], [0], color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3.5, label="Lower Safe Bound"),
            *limit_handles,
            Line2D([0], [0], color=ub_color, linewidth=3.5, label="Upper Safe Bound"),
        ]
    legend_handles = scenario_handles + limit_handles
    if show_pba:
        legend_handles += [
            Circle((0, 0), radius=0.35, facecolor="#aaa", edgecolor="#444", linewidth=1.3, label="Consumption-based"),
            Circle((0, 0), radius=0.35, facecolor="#aaa", edgecolor="#111", linewidth=1.3, hatch="///",
                   label="Production-based"),
        ]
    if show_sensitivity:
        legend_handles.append(
            Line2D([0], [0], color="#333333", linewidth=2.2, marker="|", markersize=12, markeredgewidth=2.2,
                   label="sensitivity to allocation principles")
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(8, len(legend_handles)),
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=13)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.995)
    left_rect = 0.03 if title_above_bar else 0.10
    top_rect = 0.90 if title_above_bar else 0.93
    plt.tight_layout(rect=[left_rect, 0.06, 0.98, top_rect])
    if top_main_ax is not None:
        # tight_layout() régénère les labels de l'axe principal (spine
        # déplacée plus haut) et leur fait perdre le gras appliqué via
        # set_xticklabels ; on le réapplique donc en tout dernier.
        plt.setp(top_main_ax.get_xticklabels(), fontweight="bold")

    return fig


# ============================================================================
# VARIANTE INDÉPENDANTE : create_overshoot_safe_space_figure_by_region
# ============================================================================
# Implémentation entièrement autonome : ses propres versions privées de tri des
# LP, dessin des bulles et dimensionnement des lignes (suffixe "_region"),
# distinctes de celles de create_overshoot_safe_space_figure ci-dessus, pour
# qu'une évolution propre à cette figure (ex. beaucoup de scénarios simultanés)
# ne puisse jamais changer le rendu de l'autre.

# Taille des bulles de scénario (aire, en points^2 -- paramètre `s` de
# ax.scatter), réduite par rapport à la valeur historique (300, cf. _plot_bubble
# ci-dessus) pour que les bulles ne se chevauchent plus quand une ligne a
# beaucoup de scénarios.
BUBBLE_MARKER_SIZE_REGION = 210

# Hauteur de barre minimale par scénario (mêmes unités que row_height_factor) :
# au lieu d'une hauteur de barre fixe quel que soit le nombre de scénarios (ce
# qui tasse d'autant plus les niveaux y que scenario_names est long), la
# hauteur de barre grandit avec len(scenario_names) dès que ce minimum dépasse
# la hauteur par défaut -- ce qui écarte proportionnellement les bulles et les
# étiquettes de code (affichées à gauche de chaque barre) de scénarios voisins.
# Sans effet pour le nombre de scénarios "historique" (2-4), où la hauteur par
# défaut reste plus grande que ce minimum.
MIN_ROW_HEIGHT_PER_SCENARIO_REGION = 0.30

# Plafond de row_height_factor (donc de MIN_ROW_HEIGHT_PER_SCENARIO_REGION *
# nb de scénarios) : la hauteur de figure totale est ~row_height_factor * n_lp,
# donc sans plafond, un nombre de scénarios inhabituellement grand (ex. un
# dossier data/3.10.2 contenant une ligne par pays EXIOBASE au lieu des seuls
# FR/EU27/W attendus) ferait une figure de plusieurs centaines de pouces --
# assez pour que matplotlib échoue à la rendre correctement (page blanche) au
# lieu de juste tasser les bulles. Au-delà de ce plafond, les bulles redeviennent
# tassées, mais la figure reste au moins générée.
MAX_ROW_HEIGHT_FACTOR_REGION = 3.5


def _find_region_scenario_idx(scenario_names, region_codes):
    """Index du scénario de référence (code région = config.DEFAULT_REGION_CODE,
    "FR") dans scenario_names, utilisé pour trier les LP par dépassement CBA
    (voir _sort_subprocesses_by_region_overshoot). Si aucun scénario n'a ce
    code, retombe sur config.REFERENCE_SCENARIO_IDX."""
    for idx, name in enumerate(scenario_names):
        if str(region_codes.get(name, "")).upper() == config.DEFAULT_REGION_CODE:
            return idx
    return config.REFERENCE_SCENARIO_IDX


def _union_subprocesses(all_scenarios_data):
    """Liste des sous-processus (lignes de la figure) présents dans AU MOINS UN
    scénario, dans leur ordre de première apparition.

    Ne se limite pas aux clés d'un seul scénario de référence : les scénarios
    sont chargés depuis des fichiers pouvant être partiels pour tel ou tel
    sous-processus (pipeline de données encore en cours d'écriture, LP absent
    pour une géographie...) -- prendre un seul scénario comme référence ferait
    disparaître toute la figure si c'est justement lui qui est incomplet, ou
    masquerait un LP que d'autres scénarios ont bien calculé."""
    subprocesses = []
    seen = set()
    for scenario_data in all_scenarios_data:
        for name in scenario_data.keys():
            if name not in seen:
                seen.add(name)
                subprocesses.append(name)
    return subprocesses


def _sort_subprocesses_by_region_overshoot(subprocesses, scenario_names, all_scenarios_data,
                                            budget_of, region_codes):
    """Trie les LP (lignes de la figure) par dépassement croissant du scénario
    France (max entre CBA et PBA, si PBA présent) : plus la bulle France la plus
    à droite sur une barre est loin, plus cette barre est affichée bas par
    rapport aux autres.

    budget_of(subprocess_name, scenario_name) donne le budget ("1L") auquel
    comparer l'empreinte — il dépend du scénario, chaque pays ayant sa propre
    part du budget mondial."""
    region_idx = _find_region_scenario_idx(scenario_names, region_codes)
    region_data = all_scenarios_data[region_idx]
    region_name = scenario_names[region_idx]

    def sort_key(subprocess_name):
        payload = region_data.get(subprocess_name)
        lb_val = budget_of(subprocess_name, region_name)
        if payload is None or not lb_val:
            return float("-inf")
        rel_cba = _total_footprint(payload) / lb_val
        abs_pba = payload.get("pba")
        rel_pba = abs_pba / lb_val if abs_pba is not None else float("-inf")
        rel_max = max(rel_cba, rel_pba)
        return rel_max if np.isfinite(rel_max) else float("-inf")

    return sorted(subprocesses, key=sort_key)


def _relative_bounds(abs_value, budget_min, budget_max):
    """Bornes (gauche, droite) du segment de sensibilité d'un point : overshoot de
    `abs_value` sous les deux budgets encadrants.

    Les deux résultats sont réordonnés en (min, max) car un budget plus petit
    donne un overshoot plus grand : la borne "min" d'un budget est la borne
    droite du segment, pas sa borne gauche.
    """
    if abs_value is None or not np.isfinite(abs_value):
        return None, None
    variants = [abs_value / b for b in (budget_min, budget_max) if b]
    if len(variants) < 2:
        return None, None
    return min(variants), max(variants)


def _plot_region_bubble(ax, rel_x, y_pos, style, hatched=False, hatch_color=None, value_label=None):
    """Version de _plot_bubble privée à create_overshoot_safe_space_figure_by_region
    (même logique, bulles plus petites -- BUBBLE_MARKER_SIZE_REGION -- pour rester
    lisibles quand une ligne a beaucoup de scénarios)."""
    sc = ax.scatter(rel_x, y_pos, s=BUBBLE_MARKER_SIZE_REGION, marker="o", facecolor=style["face"],
                     edgecolor=style["edge"], linewidth=1.5, zorder=2)
    if hatched:
        sc.set_hatch("///")
        if hatch_color is not None:
            sc.set_edgecolor(hatch_color)

    if value_label is not None:
        x_span = ax.get_xlim()[1] - ax.get_xlim()[0]
        txt = ax.text(rel_x + x_span * 0.015, y_pos, value_label, color=style["face"],
                       fontsize=8.2, ha="left", va="center", zorder=11, clip_on=True)
        txt.set_path_effects([pe.withStroke(linewidth=1.8, foreground="#2b2b2b")])


def create_overshoot_safe_space_figure_by_region(
    all_scenarios_data, seuils_df, shares_df, scenario_names,
    unit_row=config.UNIT_ROW_ABS,
    ub_color="#bc270a", x_main_max=8.0, display_bounds=True,
    show_pba=False, show_value_labels=True, show_value_bounds=True,
    title_above_bar=False,
    title_gap_below_bar=0.01, title_gap_above_bar=0.01,
    title="Overshoot relative to the Safe Operating Space",
    sharing_principle="EPC", sensitivity=False,
    region_codes=None, default_region_code=config.DEFAULT_REGION_CODE,
):
    """Même type de figure que create_overshoot_safe_space_figure, en valeurs
    absolues et avec un budget propre à chaque pays/région.

    Rien n'est divisé par la population : ni les empreintes (à fournir en
    valeurs absolues, cf. processing.process_scenario_absolute), ni les budgets.
    Les budgets mondiaux restent ceux de seuils.xlsx/feuille "Synthèse" ("Lower
    safe bound"/"Safe limit"/"Upper safe bound", converties par "Unit conversion
    budget" comme avant), mais la descente au niveau national se fait en les
    multipliant par la part du pays lue dans budget_shares.xlsx/feuille
    "Parts_regions_EXIOBASE", au lieu du ratio de populations :

        budget_pays = Budget_mondial / Conversion_budget * Part_pays

    Le pays/région de chaque scénario est déduit du nom de son dossier de
    data/3.10.2 via la colonne "Code EXIOBASE" ("2019_EU27" -> ligne "EU27",
    "2019_FR" -> "FR", "2019_W" -> "W") ; voir processing.resolve_region_code.

    La couleur des bulles et l'identification des scénarios ne passent pas par
    colors.scenario_style_map (couleur par motif dans le nom, utilisée par
    create_overshoot_safe_space_figure) mais par le code région résolu : "FR"
    garde le bleu clair, "EU27" le rouge, "W" le vert (colors.REGION_CODE_COLORS),
    n'importe quel autre code un gris générique -- deux scénarios de noms
    différents rattachés au même code (ex. Base_year et TREND, tous deux "FR"
    par défaut) reçoivent donc la même couleur. La légende n'a pas d'entrée par
    scénario : chaque scénario est identifié par son code, affiché en texte à
    gauche de chaque barre, aligné verticalement sur sa bulle.

    Conséquence : l'abscisse 1L n'est plus le même budget en valeur absolue pour
    tous les scénarios (chaque bulle est positionnée par rapport au budget de son
    propre pays), alors que les traits Lower/Upper Safe Bound restent, eux,
    communs à toute la ligne — leur position relative (UB/LB) ne dépend pas de la
    part, qui se simplifie dans le rapport.

    Args:
        shares_df: feuille "Parts_regions_EXIOBASE" de budget_shares.xlsx
            (io.load_budget_shares_df).
        unit_row: ligne d'unité d'affichage de seuils_df — config.UNIT_ROW_ABS
            par défaut, la figure étant en valeurs absolues.
        sharing_principle: "EPC" (equal per capita) ou "CTR" (capability to
            reduce). La part de référence, qui positionne les bulles, est lue
            dans la colonne "Part EPC ref. ..." ou "Part CTR ref. ..." selon le
            cas (config.BUDGET_SHARE_COLUMN_PATTERNS).
        sensitivity: si True, chaque bulle reçoit un segment horizontal
            (intervalle de confiance, avec petits traits verticaux aux
            extrémités) reliant les deux niveaux d'overshoot obtenus avec les
            budgets des colonnes "<principe> min" et "<principe> max". Le
            segment est blanc pour les bulles CBA (pleines) et noir pour les
            bulles PBA (hachurées, show_pba=True). Une part min donnant un
            budget plus petit, donc un overshoot plus grand, c'est elle qui
            fixe l'extrémité DROITE du segment (et la part max l'extrémité
            gauche).
        region_codes: optionnel, {nom_scénario: code EXIOBASE} pour forcer le
            rattachement d'un scénario à une ligne de shares_df, au lieu de le
            déduire de son nom.
        default_region_code: code utilisé pour un scénario dont le nom ne
            contient aucun code EXIOBASE ("FR" par défaut : Base_year, TREND,
            Tech_NZE... sont des scénarios France).

    Les autres paramètres ont exactement le même sens que dans
    create_overshoot_safe_space_figure.

    Note: show_value_bounds continue de piloter l'affichage de l'unité du LP,
    mais la valeur absolue au-dessus du trait Safe Limit n'est affichée que si
    tous les scénarios partagent le même budget (un seul pays à l'écran) —
    sinon il n'y a pas une valeur unique à annoter.

    Returns:
        matplotlib.figure.Figure
    """
    if not all_scenarios_data:
        raise ValueError("all_scenarios_data est vide")
    if sharing_principle not in config.BUDGET_SHARE_COLUMN_PATTERNS:
        raise ValueError(
            f"sharing_principle inconnu : {sharing_principle!r} "
            f"(attendu : {', '.join(config.BUDGET_SHARE_COLUMN_PATTERNS)})"
        )

    codes = {
        name: (region_codes or {}).get(
            name, processing.resolve_region_code(name, shares_df, default=default_region_code)
        )
        for name in scenario_names
    }

    def budget_of(subprocess_name, scenario_name, threshold_kind="lb", variant="ref"):
        return processing.compute_region_budget(
            seuils_df, shares_df, subprocess_name, codes[scenario_name],
            sharing_principle, variant=variant, threshold_kind=threshold_kind,
        )

    subprocesses = _union_subprocesses(all_scenarios_data)
    if not subprocesses:
        raise ValueError(
            "Aucun sous-processus trouvé dans all_scenarios_data (tous les scénarios sont vides) — "
            "vérifier que le traitement des dossiers de scénarios a bien produit des données."
        )
    subprocesses = _sort_subprocesses_by_region_overshoot(
        subprocesses, scenario_names, all_scenarios_data, budget_of, codes,
    )
    n_lp = len(subprocesses)

    scenario_style = colors.region_code_style_map(codes)
    y_levels = np.linspace(0.82, 0.18, len(scenario_names)) if len(scenario_names) > 1 else np.array([0.5])
    scenario_to_y = dict(zip(scenario_names, y_levels))

    # Hauteur de barre : la valeur par défaut (1.35/1.2), sauf si le nombre de
    # scénarios est assez grand pour l'exiger (MIN_ROW_HEIGHT_PER_SCENARIO_REGION),
    # plafonné à MAX_ROW_HEIGHT_FACTOR_REGION -- voir leurs docstrings. Sans
    # effet sur les figures à 2-4 scénarios.
    default_row_height_factor = 1.35 if title_above_bar else 1.2
    row_height_factor = max(
        default_row_height_factor,
        min(MIN_ROW_HEIGHT_PER_SCENARIO_REGION * len(scenario_names), MAX_ROW_HEIGHT_FACTOR_REGION),
    )
    # hspace dimensionné pour que le bloc de titre (nom + unité) tienne
    # toujours entre les deux écarts demandés, quels que soient leurs réglages.
    hspace = (title_gap_below_bar + title_gap_above_bar + TITLE_BLOCK_HEIGHT) if title_above_bar else 0.12
    fig = plt.figure(figsize=(18, max(8, row_height_factor * n_lp + 1.8)))
    gs = fig.add_gridspec(n_lp, 2, width_ratios=[7.0, 3.0], hspace=hspace, wspace=0.02)

    top_main_ax = None

    for row_idx, subprocess_name in enumerate(subprocesses):
        ax = fig.add_subplot(gs[row_idx, 0])
        ax_ext = fig.add_subplot(gs[row_idx, 1], sharey=ax)
        ax.set_zorder(3)
        ax_ext.set_zorder(2)
        ax_ext.patch.set_alpha(0.0)
        if row_idx == 0:
            top_main_ax = ax

        unit_text = _get_lp_unit(seuils_df, subprocess_name, unit_row) if show_value_bounds else ""

        # Un budget par scénario (= par pays) : le budget de référence sert à
        # positionner les bulles de ce scénario, et les budgets min/max à
        # encadrer le segment de sensibilité.
        budgets = {name: budget_of(subprocess_name, name) for name in scenario_names}
        if all(b is None for b in budgets.values()):
            ax.text(0.5, 0.5, f"{subprocess_name}: limite basse manquante", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#7a7a7a")
            ax.set_xlim(0, x_main_max)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax_ext.set_visible(False)
            continue

        # Calcul des positions relatives (CBA, et PBA si demandé) par scénario
        points = []
        for scenario_idx, scenario_data in enumerate(all_scenarios_data):
            sname = scenario_names[scenario_idx]
            lb_val = budgets[sname]
            if subprocess_name not in scenario_data or not lb_val:
                continue
            payload = scenario_data[subprocess_name]
            abs_cba = _total_footprint(payload)
            rel_cba = abs_cba / lb_val
            if not np.isfinite(rel_cba):
                continue

            abs_pba = payload.get("pba") if show_pba else None
            rel_pba = abs_pba / lb_val if (abs_pba is not None and np.isfinite(abs_pba)) else None

            rel_cba_min = rel_cba_max = None
            rel_pba_min = rel_pba_max = None
            if sensitivity:
                budget_min = budget_of(subprocess_name, sname, variant="min")
                budget_max = budget_of(subprocess_name, sname, variant="max")
                rel_cba_min, rel_cba_max = _relative_bounds(abs_cba, budget_min, budget_max)
                if abs_pba is not None:
                    rel_pba_min, rel_pba_max = _relative_bounds(abs_pba, budget_min, budget_max)

            points.append({
                "sname": sname, "y_pos": scenario_to_y[sname],
                "abs_cba": abs_cba, "rel_cba": rel_cba,
                "abs_pba": abs_pba, "rel_pba": rel_pba,
                "rel_cba_min": rel_cba_min, "rel_cba_max": rel_cba_max,
                "rel_pba_min": rel_pba_min, "rel_pba_max": rel_pba_max,
            })

        # Dépassement d'axe : agrège TOUTES les valeurs x présentes sur cette
        # ligne (pas juste cba/pba en dur), pour rester correct si une autre
        # série s'ajoute un jour.
        all_x = [p["rel_cba"] for p in points] + [p["rel_pba"] for p in points if p["rel_pba"] is not None]
        overflow = [x for x in all_x if x > x_main_max]
        use_break = len(overflow) > 0

        ax.set_xlim(0, x_main_max)
        ax.set_ylim(0, 1)
        _draw_lp_background(ax, 0, x_main_max)

        if use_break:
            x2_min = min(overflow) * 0.995
            x2_max = max(overflow) * 1.02
            if (x2_max - x2_min) < 0.5:
                mid = 0.5 * (x2_min + x2_max)
                x2_min, x2_max = mid - 0.25, mid + 0.25

            ax_ext.set_xlim(x2_min, x2_max)
            ax_ext.set_ylim(0, 1)
            _draw_above_break_uniform(ax_ext, x2_min, x2_max)
            _set_break_axis_ticks(ax_ext)

            d = 0.012
            kw_main = dict(transform=ax.transAxes, color="#3a3a3a", clip_on=False, linewidth=1.2)
            kw_ext = dict(transform=ax_ext.transAxes, color="#3a3a3a", clip_on=False, linewidth=1.2)
            ax.plot((1 - d, 1 + d), (-d, +d), **kw_main)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw_main)
            ax_ext.plot((-d, +d), (-d, +d), **kw_ext)
            ax_ext.plot((-d, +d), (1 - d, 1 + d), **kw_ext)
        else:
            ax_ext.set_visible(False)

        # Les traits Lower/Upper Safe Bound sont communs à la ligne : leur
        # position relative (Budget_mondial_LOWER/UB / Budget_mondial_LB) est la
        # même pour tous les pays, la part se simplifiant dans le rapport. On
        # les calcule donc avec le budget du scénario de référence.
        ref_name = next(name for name in scenario_names if budgets[name])
        ref_lb = budgets[ref_name]
        lower_val = budget_of(subprocess_name, ref_name, threshold_kind="lower")
        ub_val = budget_of(subprocess_name, ref_name, threshold_kind="ub")
        lower_rel = (lower_val / ref_lb) if lower_val is not None else None
        ub_rel = (ub_val / ref_lb) if ub_val is not None else None
        distinct_budgets = {round(b, 12) for b in budgets.values() if b}
        lb_label = _fmt_abs(ref_lb) if (show_value_bounds and len(distinct_budgets) == 1) else None

        # Traits + valeurs des limites (indépendants du dégradé de fond) : Safe
        # Limit (vert, seuil historique) toujours affiché, valeur visible
        # uniquement si un budget unique est partagé par tous les scénarios ;
        # Lower/Upper Safe Bound (vert clair/rouge) uniquement si
        # display_bounds=True, jamais leur valeur.
        if display_bounds and lower_rel is not None and lower_rel >= 0:
            ax.axvline(lower_rel, ymin=0.10, ymax=0.90, color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3, zorder=3)
        ax.axvline(1, ymin=0.10, ymax=0.90, color="#0d9a33", linewidth=3, zorder=3)
        if lb_label is not None:
            ax.text(1, 0.93, lb_label, color="#0d9a33", fontsize=11, fontweight="bold",
                    ha="center", va="bottom", zorder=5)
        if display_bounds and ub_rel is not None and ub_rel <= x_main_max:
            ax.axvline(ub_rel, ymin=0.10, ymax=0.90, color=ub_color, linewidth=3, zorder=3)

        # Bulles CBA (pleines) + PBA (hachurées, si show_pba), avec segment de
        # sensibilité (sensitivity=True) dessiné sous chaque bulle.
        for p in points:
            style = scenario_style.get(p["sname"], colors.REGION_CODE_FALLBACK)

            in_ext_cba = use_break and p["rel_cba"] > x_main_max and ax_ext.get_visible()
            target_ax_cba = ax_ext if in_ext_cba else ax
            if p["rel_cba_min"] is not None:
                _plot_sensitivity_segment(target_ax_cba, p["rel_cba_min"], p["rel_cba_max"], p["y_pos"],
                                           color="white")
            _plot_region_bubble(target_ax_cba, p["rel_cba"], p["y_pos"], style, hatched=False,
                                 value_label=_fmt_abs(p["abs_cba"]) if show_value_labels else None)

            if p["rel_pba"] is not None:
                hatch_color = "white" if "trend" in p["sname"].lower() else "black"
                in_ext_pba = use_break and p["rel_pba"] > x_main_max and ax_ext.get_visible()
                target_ax_pba = ax_ext if in_ext_pba else ax
                if p["rel_pba_min"] is not None:
                    _plot_sensitivity_segment(target_ax_pba, p["rel_pba_min"], p["rel_pba_max"], p["y_pos"],
                                               color="black")
                _plot_region_bubble(target_ax_pba, p["rel_pba"], p["y_pos"], style, hatched=True,
                                     hatch_color=hatch_color,
                                     value_label=_fmt_abs(p["abs_pba"]) if show_value_labels else None)

        # Étiquettes de code à gauche de la barre : remplace la légende "1
        # entrée par scénario" par un court libellé (le code région) aligné sur
        # la bulle de chaque scénario, répété sur chaque ligne puisque la
        # position en y d'un scénario est commune à toute la figure. x en
        # fraction d'axes, y en coordonnées de données (get_yaxis_transform),
        # pour rester juste à gauche de x=0 quel que soit x_main_max.
        yaxis_transform = ax.get_yaxis_transform()
        for sname, y_pos in scenario_to_y.items():
            style = scenario_style.get(sname, colors.REGION_CODE_FALLBACK)
            code_text = str(codes.get(sname, sname))
            ax.text(-0.012, y_pos, code_text, transform=yaxis_transform, ha="right", va="center",
                    fontsize=11, fontweight="bold", color=style["face"], clip_on=False)

        # Nom du LP + unité : dans la marge de gauche (défaut), ou au-dessus
        # de la barre (title_above_bar=True). `ceiling` (= 1.0 + hspace, le bas
        # de la barre du dessus) est la même formule pour toutes les lignes, y
        # compris la 1ère : hspace est dimensionné (cf. plus haut) pour laisser
        # aussi la place aux graduations de l'axe X partagé au-dessus d'elle.
        if title_above_bar:
            _draw_lp_title_above(ax, subprocess_name, unit_text, 1.0 + hspace,
                                  title_gap_below_bar, title_gap_above_bar)
        else:
            ax.text(-0.03, 0.57, subprocess_name, transform=ax.transAxes, ha="right", va="center",
                    fontsize=13, fontweight="bold")
            if unit_text:
                ax.text(-0.03, 0.39, f"({unit_text})", transform=ax.transAxes, ha="right", va="center",
                        fontsize=13, color="#4a4a4a")

        # Habillage épuré
        for axis in (ax, ax_ext):
            if not axis.get_visible():
                continue
            axis.set_yticks([])
            axis.grid(False)
            for spine in ("left", "right", "bottom"):
                axis.spines[spine].set_visible(False)
            axis.spines["top"].set_visible((axis is ax and row_idx == 0) or (axis is ax_ext and use_break))
            axis.xaxis.tick_top()
            axis.tick_params(axis="x", length=4, pad=4)

        if row_idx != 0:
            ax.tick_params(axis="x", labeltop=False, top=False)
        if use_break and ax_ext.get_visible():
            ax_ext.tick_params(axis="x", labeltop=True, top=True, labelsize=AXIS_TICK_LABEL_FONTSIZE, length=3, pad=2)

    # Axe X principal commun (1L, 2L, 3L...) + labels de zone, sur la 1ère ligne
    if top_main_ax is not None:
        top_main_ax.set_xticks(np.arange(1, int(x_main_max) + 1))
        top_main_ax.set_xticklabels([f"{i}L" for i in range(1, int(x_main_max) + 1)], fontsize=AXIS_TICK_LABEL_FONTSIZE, fontweight="bold")
        top_spine_y = (1.0 + hspace + 0.03) if title_above_bar else 1.03
        top_main_ax.spines["top"].set_position(("axes", top_spine_y))
        top_main_ax.tick_params(axis="x", pad=8, length=5, labelsize=AXIS_TICK_LABEL_FONTSIZE)
        zone_label_y = (top_spine_y + 0.45) if title_above_bar else 1.44
        for txt, color, x_pos in [
            ("Safe Operating Space", "#0b7d3e", 0.00),
            ("Increasing Risk", "#d47818", 0.20),
            ("High-Risk Zone", "#7a1f54", 0.40),
        ]:
            top_main_ax.text(x_pos, zone_label_y, txt, transform=top_main_ax.transAxes, color=color,
                              fontsize=11, fontweight="bold", ha="left")

    # Légende : limites + type de comptabilité (show_pba) + sensibilité
    # (sensitivity) -- pas d'entrée par scénario, chaque scénario est identifié
    # par son code à gauche de chaque barre (cf. plus haut).
    limit_handles = [Line2D([0], [0], color="#0d9a33", linewidth=3.5, label="Safe Limit")]
    if display_bounds:
        limit_handles = [
            Line2D([0], [0], color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3.5, label="Lower Safe Bound"),
            *limit_handles,
            Line2D([0], [0], color=ub_color, linewidth=3.5, label="Upper Safe Bound"),
        ]
    legend_handles = list(limit_handles)
    if show_pba:
        legend_handles += [
            Circle((0, 0), radius=0.35, facecolor="#aaa", edgecolor="#444", linewidth=1.3, label="Consumption-based"),
            Circle((0, 0), radius=0.35, facecolor="#aaa", edgecolor="#111", linewidth=1.3, hatch="///",
                   label="Production-based"),
        ]
    if sensitivity:
        legend_handles.append(
            Line2D([0], [0], color="#333333", linewidth=2.2, marker="|", markersize=12, markeredgewidth=2.2,
                   label="sensitivity to allocation principles")
        )

    fig.legend(handles=legend_handles, loc="lower center", ncol=min(8, len(legend_handles)),
               bbox_to_anchor=(0.5, -0.01), frameon=False, fontsize=13)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.995)
    left_rect = 0.03 if title_above_bar else 0.10
    top_rect = 0.90 if title_above_bar else 0.93
    plt.tight_layout(rect=[left_rect, 0.06, 0.98, top_rect])
    if top_main_ax is not None:
        # tight_layout() régénère les labels de l'axe principal (spine
        # déplacée plus haut) et leur fait perdre le gras appliqué via
        # set_xticklabels ; on le réapplique donc en tout dernier.
        plt.setp(top_main_ax.get_xticklabels(), fontweight="bold")

    return fig
