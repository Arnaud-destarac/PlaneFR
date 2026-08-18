"""
Figure "Overshoot / Safe Operating Space" : 1 ligne par limite planétaire (LP),
position de chaque scénario en abscisse relative à la limite basse (1L = seuil
d'égalité), fond dégradé vert -> orange -> violet (zone sûre -> risque
croissant -> zone à haut risque), rupture d'axe si un scénario dépasse
x_main_max fois la limite basse.

Une seule fonction sert les deux usages du projet :
  - Overshoot multi-scénarios : valeurs absolues, 1 bulle par scénario (CBA
    uniquement, pas de comptabilité PBA calculée pour ce cas).
  - Figure comparaison : valeurs par habitant, 2 bulles par scénario (CBA
    pleine + PBA hachurée).

Par décision explicite : Figure comparaison.ipynb est la référence pour les
valeurs par défaut (couleur de limite haute, absence d'étiquette de valeur sur
les bulles, scenario_style_map) — voir planefr_lib.colors.scenario_style_map
et le README du refactor pour le détail des divergences résolues.
"""

import re

import numpy as np
import pandas as pd
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
# HELPERS DE FORMATAGE ET DE CALCUL
# ============================================================================


def _fmt_abs(value):
    """Formate une valeur absolue pour annotation (séparateur de milliers insécable)."""
    if value is None or not np.isfinite(value):
        return "NA"
    if abs(value) >= 1000:
        return f"{value:,.0f}".replace(",", " ")
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


# ============================================================================
# DESSIN DU FOND ET DE L'AXE CASSÉ
# ============================================================================


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
# FIGURE PRINCIPALE
# ============================================================================


def create_overshoot_safe_space_figure(
    all_scenarios_data, seuils_df, scenario_names,
    threshold_lb_row, threshold_ub_row, unit_row,
    ub_color="#bc270a", x_main_max=8.0, display_bounds=True,
    show_pba=False, show_value_labels=True,
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
        pop_df: feuille "Population" de seuils.xlsx (io.load_population_df) --
            requis seulement si sharing_principle est fourni.

    Returns:
        matplotlib.figure.Figure
    """
    if not all_scenarios_data:
        raise ValueError("all_scenarios_data est vide")

    first_scenario_data = all_scenarios_data[config.REFERENCE_SCENARIO_IDX]
    subprocesses = list(first_scenario_data.keys())
    subprocesses = _sort_subprocesses_by_france_overshoot(
        subprocesses, scenario_names, all_scenarios_data, seuils_df,
        threshold_lb_row, pop_df, sharing_principle,
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

        unit_text = _get_lp_unit(seuils_df, subprocess_name, unit_row)
        lb_val = processing.lookup_threshold(seuils_df, threshold_lb_row, subprocess_name,
                                              pop_df=pop_df, sharing_principle=sharing_principle)
        ub_val = processing.lookup_threshold(seuils_df, threshold_ub_row, subprocess_name,
                                              pop_df=pop_df, sharing_principle=sharing_principle)
        threshold_lower_row = config.THRESHOLD_LOWER_FOR.get(threshold_lb_row)
        lower_val = (
            processing.lookup_threshold(seuils_df, threshold_lower_row, subprocess_name,
                                         pop_df=pop_df, sharing_principle=sharing_principle)
            if threshold_lower_row else None
        )

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

            sname = scenario_names[scenario_idx]
            points.append({
                "sname": sname, "y_pos": scenario_to_y[sname],
                "abs_cba": abs_cba, "rel_cba": rel_cba,
                "abs_pba": abs_pba, "rel_pba": rel_pba,
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
        # Limit (vert, seuil historique) toujours affiché ; Lower/Upper Safe
        # Bound (vert clair/rouge) uniquement si display_bounds=True.
        if display_bounds and lower_rel is not None and lower_rel >= 0:
            ax.axvline(lower_rel, ymin=0.10, ymax=0.90, color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3, zorder=3)
            ax.text(lower_rel, 0.93, _fmt_abs(lower_val), color=config.LOWER_SAFE_BOUND_COLOR, fontsize=11,
                    fontweight="bold", ha="center", va="bottom", zorder=5)
        ax.axvline(1, ymin=0.10, ymax=0.90, color="#0d9a33", linewidth=3, zorder=3)
        ax.text(1, 0.93, _fmt_abs(lb_val), color="#0d9a33", fontsize=11, fontweight="bold",
                ha="center", va="bottom", zorder=5)
        if display_bounds and ub_rel is not None and ub_rel <= x_main_max:
            ax.axvline(ub_rel, ymin=0.10, ymax=0.90, color=ub_color, linewidth=3, zorder=3)
            ax.text(ub_rel, 0.93, _fmt_abs(ub_val), color=ub_color, fontsize=11, fontweight="bold",
                    ha="center", va="bottom", zorder=5)

        # Bulles CBA (pleines) + PBA (hachurées, si show_pba)
        for p in points:
            style = scenario_style.get(p["sname"], {"face": "#b7c4d3", "edge": "#44505c"})

            in_ext_cba = use_break and p["rel_cba"] > x_main_max and ax_ext.get_visible()
            _plot_bubble(ax_ext if in_ext_cba else ax, p["rel_cba"], p["y_pos"], style, hatched=False,
                         value_label=_fmt_abs(p["abs_cba"]) if show_value_labels else None)

            if p["rel_pba"] is not None:
                hatch_color = "white" if "trend" in p["sname"].lower() else "black"
                in_ext_pba = use_break and p["rel_pba"] > x_main_max and ax_ext.get_visible()
                _plot_bubble(ax_ext if in_ext_pba else ax, p["rel_pba"], p["y_pos"], style, hatched=True,
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
        top_main_ax.spines["top"].set_position(("outward", 8))
        # labelsize (contrairement à fontsize passé à set_xticklabels) reste
        # appliqué même si l'axe régénère ses labels suite au déplacement de
        # spine ci-dessus + à plt.tight_layout() en fin de fonction.
        top_main_ax.tick_params(axis="x", pad=8, length=5, labelsize=AXIS_TICK_LABEL_FONTSIZE)
        zone_label_y = (1.0 + hspace + 0.15) if title_above_bar else 1.44
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
