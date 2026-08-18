"""
Barres empilées par sous-processus (create_stacked_bar_chart) et deux figures de
synthèse multi-scénarios qui l'assemblent en grille :
  - create_synthesis_figure        : normalisée en %, 1 grille 2 colonnes
  - create_synthesis_figure_by_lp  : valeurs absolues, grille 3x3 par LP, avec
                                      seuils (LP/DLS) et rupture d'axe si besoin

N'est utilisé que par les figures en valeurs absolues (Synthèse multi-scénarios) :
les constantes de seuils utilisées sont donc fixées à la variante "_ABS" de
planefr_lib.config plutôt que passées en paramètre (contrairement à
plot_overshoot.create_overshoot_safe_space_figure, partagée avec la variante
par habitant).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from . import colors, config, processing

CATEGORY_LEGEND_LABELS = {"F_Y": "Not allocated"}


def _build_category_legend(categories):
    """Légende commune aux figures de synthèse : un carré (teinte foncée) par
    catégorie de consommation, suivi d'un unique motif hachuré "Import".
    F_Y (émissions directes non réparties par la bridge) est affiché sous le
    nom "Not allocated".
    """
    handles, labels = [], []
    for cat_idx, category in enumerate(categories):
        color_dark = colors.get_dark_shade(colors.get_category_color(category, cat_idx))
        handles.append(Patch(facecolor=color_dark, edgecolor="white"))
        labels.append(CATEGORY_LEGEND_LABELS.get(category, category))

    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=1))
    labels.append("Import")
    return handles, labels


def create_stacked_bar_chart(data_by_subprocess, seuils_df, subprocess_to_lp,
                              scenario_name="", reference_data=None, ax=None, show_legend=True,
                              group_by_category=True, sharing_principle=None, pop_df=None):
    """Barres empilées en % représentant les LP par sous-processus : dom (plein) +
    imp (hachuré) empilés par catégorie de consommation. Annote le total absolu,
    la variation relative au scénario de référence (si fourni), et le dépassement
    du seuil bas.

    Args:
        data_by_subprocess: {nom_sous_processus: résultat de process_single_subprocess_scenario}
        subprocess_to_lp: mapping sous-processus -> [LP] (non utilisé directement ici,
            gardé en signature pour cohérence avec les appelants)
        reference_data: data_by_subprocess d'un scénario de référence, pour afficher
            la variation relative (omettre pour le scénario de référence lui-même)
        ax: axe matplotlib existant ; si None, une nouvelle figure est créée
        group_by_category: True (défaut) pour empiler dom/imp l'un après l'autre pour
            chaque catégorie (cat1-dom, cat1-imp, cat2-dom, cat2-imp, ...) ; False pour
            regrouper par origine (toutes les catégories dom, puis toutes les catégories imp)
        sharing_principle: si fourni ("Equality 2050"/"Equality 2100"/"Equality 2019"),
            le seuil bas (LB) est recalculé à la volée par un partage égal per capita du
            budget mondial pour la période de référence associée (voir
            config.SHARING_PRINCIPLE_POPULATION_ROW et processing.compute_sharing_seuil),
            au lieu d'être lu statiquement dans seuils_df. Nécessite pop_df.
        pop_df: feuille "Population" de seuils.xlsx (io.load_population_df) -- requis
            seulement si sharing_principle est fourni.

    Returns:
        (fig, ax) si ax=None a été fourni en entrée, sinon (None, ax).
    """
    subprocesses = list(data_by_subprocess.keys())
    n_subprocesses = len(subprocesses)

    if not any(data is not None for data in data_by_subprocess.values()):
        print("Aucune donnée valide à visualiser")
        return None if ax is None else (None, ax)

    all_categories = set()
    for data in data_by_subprocess.values():
        if data is not None:
            all_categories.update(data["categories"])
    categories = colors.sort_categories(list(all_categories))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = None

    def heights_for(key, category):
        return np.array([
            data_by_subprocess[sp][key].get(category, 0)
            if (data_by_subprocess[sp] is not None and key in data_by_subprocess[sp]) else 0
            for sp in subprocesses
        ])

    bar_width = 0.6
    x_pos = np.arange(n_subprocesses)

    total_values = np.zeros(n_subprocesses)
    for category in categories:
        total_values += heights_for("domestique", category) + heights_for("importé", category)

    def draw_segment(category, cat_idx, key, bottom_heights, hatch=None):
        color_dark = colors.get_dark_shade(colors.get_category_color(category, cat_idx))
        label_suffix = "Import" if hatch else "Domestic"
        heights_pct = np.where(total_values > 0, heights_for(key, category) / total_values * 100, 0)
        ax.bar(x_pos, heights_pct, bar_width, label=f"{category} - {label_suffix}",
               bottom=bottom_heights, color=color_dark, edgecolor="white", linewidth=0.5, hatch=hatch)
        return heights_pct

    bottom_heights = np.zeros(n_subprocesses)
    if group_by_category:
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, "domestique", bottom_heights)
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, "importé", bottom_heights, hatch="/")
    else:
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, "domestique", bottom_heights)
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, "importé", bottom_heights, hatch="/")

    for idx, total_val in enumerate(total_values):
        sp = subprocesses[idx]

        unit_val = processing.lookup_first_text(seuils_df, config.UNIT_ROW_ABS, sp)
        ax.text(idx, 102, f"{total_val:.0f}{f' {unit_val}' if unit_val else ''}",
                ha="center", va="bottom", fontweight="bold", fontsize=10.5)

        if reference_data is not None and sp in reference_data:
            ref_total = reference_data[sp]["domestique"].sum() + reference_data[sp]["importé"].sum()
            if ref_total > 0:
                variation_pct = (total_val - ref_total) / ref_total * 100
                text_var = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
                ax.text(idx, 97, text_var, ha="center", va="bottom", fontweight="bold", fontsize=11, color="black")

        threshold = processing.lookup_threshold(seuils_df, config.THRESHOLD_LB_ABS, sp,
                                                 pop_df=pop_df, sharing_principle=sharing_principle)
        if threshold is not None:
            overshoot_pct = total_val / threshold
            color = "red" if overshoot_pct > 1 else "lightgreen"
            y_pos = 95 if reference_data is not None else 98
            ax.text(idx, y_pos, f"{overshoot_pct:.1f}", ha="center", va="top",
                    fontweight="bold", fontsize=14, color=color)

    ax.set_ylabel("Footprint share (%)", fontsize=12, fontweight="bold")
    if scenario_name:
        ax.set_title(f"{scenario_name} - Consumption-based", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subprocesses, rotation=30, ha="center", fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if show_legend:
        handles, labels = _build_category_legend(categories)
        handles.append(Line2D([0], [0], marker="$2.5$", color="red", linestyle="None", markersize=13))
        labels.append("Overshoot")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=13)

    if fig is not None:
        plt.tight_layout()

    return (fig, ax) if fig is not None else (None, ax)


def create_synthesis_figure(all_scenarios_data, seuils_df, subprocess_to_lp_list, scenario_names,
                             group_by_category=True, sharing_principle=None, pop_df=None):
    """Grille de create_stacked_bar_chart pour tous les scénarios (2 colonnes,
    autant de lignes que nécessaire), avec légende et titre communs.

    Args:
        group_by_category: voir create_stacked_bar_chart
        sharing_principle, pop_df: voir create_stacked_bar_chart

    Returns:
        (fig, axes_list)
    """
    n_scenarios = len(all_scenarios_data)
    n_cols = 2 if n_scenarios > 1 else 1
    n_rows = int(np.ceil(n_scenarios / n_cols))

    fig = plt.figure(figsize=(9 * n_cols, 4 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    reference_idx = config.REFERENCE_SCENARIO_IDX
    reference_data = all_scenarios_data[reference_idx]
    positions = [(i // n_cols, i % n_cols) for i in range(n_scenarios)]

    axes_list = []

    for scenario_idx, (row, col) in enumerate(positions):
        ax = fig.add_subplot(gs[row, col])
        axes_list.append(ax)

        is_reference = scenario_idx == reference_idx
        _, ax = create_stacked_bar_chart(
            all_scenarios_data[scenario_idx], seuils_df, subprocess_to_lp_list[scenario_idx],
            scenario_name=scenario_names[scenario_idx],
            reference_data=None if is_reference else reference_data,
            ax=ax, show_legend=False, group_by_category=group_by_category,
            sharing_principle=sharing_principle, pop_df=pop_df,
        )

    all_categories = set()
    for scenario_data in all_scenarios_data:
        for data in scenario_data.values():
            if data is not None:
                all_categories.update(data["categories"])
    all_handles, all_labels = _build_category_legend(colors.sort_categories(list(all_categories)))

    fig.legend(all_handles, all_labels, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fancybox=True, shadow=True)

    fig.suptitle(
        "Décomposition sectorielle des empreintes environnementales - Comparaison des scénarios"
        f"(Critère d'équité: {config.THRESHOLD_LB_ABS})",
        fontsize=16, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes_list


def create_synthesis_figure_by_lp(all_scenarios_data, seuils_df, subprocess_to_lp_list, scenario_names,
                                   dls_df=None, group_by_category=True, show_pba=True, show_dls=True,
                                   show_values=True, show_thresholds=True, display_bounds=True,
                                   show_variation_bars=False,
                                   sharing_principle=None, pop_df=None):
    """Figure de synthèse alternative : 7 LP en grille 3x3 (1 barre par scénario,
    1 subplot par LP), valeurs absolues (non normalisées). Affiche le plafond
    environnemental (Safe Limit, Upper Safe Bound) et le plancher social (DLS)
    s'ils existent, avec rupture d'axe automatique quand le plafond haut dépasse
    largement le maximum observé.

    Args:
        group_by_category: voir create_stacked_bar_chart
        show_pba: True (défaut) pour afficher la croix production-based (PBA)
            et ses annotations sur chaque barre ; False pour les masquer
            entièrement (barres, texte, légende).
        show_dls: True (défaut) pour afficher le plancher social (DLS) quand
            dls_df fournit une valeur pour le LP ; False pour l'ignorer même
            si dls_df est fourni.
        show_values: True (défaut) pour afficher les valeurs/variations
            au-dessus des barres (CBA et PBA) ; False les masque (les barres,
            la croix PBA et les traits de seuil restent inchangés).
        show_thresholds: True (défaut) pour afficher les traits horizontaux de
            seuil (Lower Safe Bound, Safe Limit, Upper Safe Bound, DLS) et la
            rupture d'axe qui leur est associée ; False les masque (les valeurs
            au-dessus des barres restent pilotées indépendamment par show_values).
        display_bounds: True (défaut, pertinent seulement si show_thresholds=True)
            affiche Lower Safe Bound et Upper Safe Bound en plus de Safe Limit
            (toujours affiché). False les masque (seul Safe Limit -- et DLS si
            show_dls -- reste affiché).
        show_variation_bars: False (défaut) pour l'affichage habituel (chaque
            barre = empreinte absolue empilée par catégorie). True pour afficher,
            pour tous les scénarios sauf la référence, uniquement la variation
            par rapport à la référence : chaque segment catégorie/origine
            (domestique ou importé) est empilé au-dessus du total de la
            référence (l'"ordonnée 0" de la vue) s'il augmente, ou empilé
            en-dessous s'il diminue, avec une hauteur égale à la valeur absolue
            de la variation. La barre de la référence reste inchangée (empilée
            depuis 0) et sert de repère visuel pour ce niveau zéro (trait
            pointillé). Une croix marque, pour chaque scénario, le total réel
            de son empreinte (utile quand des segments montent et d'autres
            descendent, pour voir où se situe le solde net).
        sharing_principle, pop_df: voir create_stacked_bar_chart -- s'appliquent
            ici à threshold_lower, threshold_lb ET threshold_ub.

    Returns:
        (fig, axes_list)
    """
    reference_idx = config.REFERENCE_SCENARIO_IDX
    # Le scénario de référence (base_year) garde sa position ; les autres sont
    # affichés dans l'ordre inverse (celui qui était le plus loin dans
    # l'alphabet -- donc le plus éloigné de la référence -- se retrouve
    # désormais juste à côté d'elle, et inversement) -- voir la même logique
    # dans create_radial_synthesis_figure.
    other_indices = [i for i in range(len(scenario_names)) if i != reference_idx]
    display_order = list(range(len(scenario_names)))
    for slot, idx in zip(other_indices, reversed(other_indices)):
        display_order[slot] = idx
    scenario_names = [scenario_names[i] for i in display_order]
    all_scenarios_data = [all_scenarios_data[i] for i in display_order]
    subprocess_to_lp_list = [subprocess_to_lp_list[i] for i in display_order]

    first_scenario_data = all_scenarios_data[reference_idx]
    subprocesses = list(first_scenario_data.keys())
    n_lp = len(subprocesses)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.15, top=0.93, bottom=0.12, left=0.045, right=0.99)
    positions = [(r, c) for r in range(3) for c in range(3)]

    reference_idx = config.REFERENCE_SCENARIO_IDX
    axes_list = []
    legend_categories = set()

    def _finalize_subplot(ax, ax_top, use_break, top, pba_values, total_values):
        """Deuxième passe (après alignement du 0 % sur la ligne) : fixe la
        limite haute de l'axe puis dessine tout ce qui dépend de ax.get_ylim()
        (croix PBA, marqueur de total en mode variation, textes de valeurs).
        """
        ax.set_ylim(0, top)

        # PBA (production-based accounting) : croix noire en gras sur chaque barre,
        # à comparer visuellement au sommet de la barre (CBA, consumption-based).
        pba_axes = [None] * len(pba_values)
        for scenario_idx, pba_val in enumerate(pba_values):
            if pba_val is None or not np.isfinite(pba_val):
                continue
            target_ax = ax_top if (use_break and ax_top is not None and pba_val > ax.get_ylim()[1]) else ax
            pba_axes[scenario_idx] = target_ax
            target_ax.scatter(scenario_idx, pba_val, marker="X", s=140, color="black",
                               linewidths=1.5, zorder=12)

        # Mode variation : croix "+" marquant, pour chaque scénario, le total réel
        # de son empreinte (CBA) -- utile car le solde net d'un scénario ne
        # correspond pas forcément au sommet des segments empilés vers le haut
        # quand certains segments montent et d'autres descendent.
        if show_variation_bars:
            for scenario_idx, total_val in enumerate(total_values):
                target_ax = ax_top if (use_break and ax_top is not None and total_val > ax.get_ylim()[1]) else ax
                target_ax.scatter(scenario_idx, total_val, marker="P", s=170, color="black",
                                   edgecolors="white", linewidths=1, zorder=14)

        if show_values:
            ref_total = total_values[reference_idx]
            ref_pba = pba_values[reference_idx]
            ax_margin = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.02
            # Superposition jugée réelle seulement si la croix PBA est sur le même
            # axe que la barre (pas de conflit possible si elle est sur ax_top, au-
            # delà de la rupture d'axe) ET proche du total en distance absolue sur
            # l'axe.
            overlap_threshold = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.1

            for scenario_idx, total_val in enumerate(total_values):
                pba_val = pba_values[scenario_idx]
                pba_ax = pba_axes[scenario_idx]
                overlapping = (
                    pba_ax is ax and pba_val is not None and np.isfinite(pba_val)
                    and abs(pba_val - total_val) < overlap_threshold
                )
                cba_y, cba_va = (pba_val - ax_margin, "top") if overlapping else (total_val + ax_margin, "bottom")

                if scenario_idx == reference_idx:
                    ax.text(scenario_idx, cba_y, f"{total_val:.0f}",
                            ha="center", va=cba_va, fontweight="bold", fontsize=15, color="black", zorder=13)
                elif ref_total > 0:
                    variation_pct = (total_val - ref_total) / ref_total * 100
                    text_var = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
                    var_color = "red" if variation_pct > 0 else ("green" if variation_pct < 0 else "black")
                    ax.text(scenario_idx, cba_y, text_var,
                            ha="center", va=cba_va, fontweight="bold", fontsize=15, color=var_color, zorder=13)

                # Texte PBA (total pour la référence, variation relative à la
                # référence pour les autres), toujours au-dessus de la croix.
                if pba_val is None or not np.isfinite(pba_val) or pba_ax is None:
                    continue
                pba_margin = (pba_ax.get_ylim()[1] - pba_ax.get_ylim()[0]) * 0.02
                pba_y = pba_val + pba_margin

                if scenario_idx == reference_idx:
                    pba_ax.text(scenario_idx, pba_y, f"{pba_val:.0f}",
                                ha="center", va="bottom", fontweight="bold", style="italic", fontsize=15, color="black", zorder=13)
                elif ref_pba is not None and np.isfinite(ref_pba) and ref_pba > 0:
                    pba_variation_pct = (pba_val - ref_pba) / ref_pba * 100
                    pba_text_var = f"+{pba_variation_pct:.0f}%" if pba_variation_pct > 0 else f"{pba_variation_pct:.0f}%"
                    pba_var_color = "#9c0000" if pba_variation_pct > 0 else ("#005e00" if pba_variation_pct < 0 else "black")
                    pba_ax.text(scenario_idx, pba_y, pba_text_var,
                                ha="center", va="bottom", fontweight="bold", style="italic", fontsize=15, color=pba_var_color, zorder=13)

    row_items = []

    for lp_idx, subprocess_name in enumerate(subprocesses):
        if lp_idx >= len(positions):
            break

        row, col = positions[lp_idx]
        ax = fig.add_subplot(gs[row, col])
        axes_list.append(ax)

        n_scenarios = len(all_scenarios_data)
        bar_width = 0.6
        x_pos = np.arange(n_scenarios)

        categories = first_scenario_data[subprocess_name]["categories"]
        legend_categories.update(categories)

        def heights_for(scenario_data, key, category):
            if subprocess_name not in scenario_data:
                return 0
            return scenario_data[subprocess_name][key].get(category, 0)

        # PREMIÈRE PASSE : totaux absolus par scénario (pour dimensionner l'axe Y)
        total_values = np.zeros(n_scenarios)
        for category in categories:
            for scenario_idx, scenario_data in enumerate(all_scenarios_data):
                total_values[scenario_idx] += (
                    heights_for(scenario_data, "domestique", category)
                    + heights_for(scenario_data, "importé", category)
                )

        max_total = float(np.max(total_values)) if total_values.size else 0.0

        if show_pba:
            pba_values = [
                scenario_data[subprocess_name].get("pba") if subprocess_name in scenario_data else None
                for scenario_data in all_scenarios_data
            ]
            finite_pba = [float(v) for v in pba_values if v is not None and np.isfinite(v)]
            max_pba = max(finite_pba) if finite_pba else 0.0
        else:
            pba_values = [None] * n_scenarios
            max_pba = 0.0

        threshold_lb = processing.lookup_threshold(seuils_df, config.THRESHOLD_LB_ABS, subprocess_name,
                                                    pop_df=pop_df, sharing_principle=sharing_principle)
        threshold_ub = processing.lookup_threshold(seuils_df, config.THRESHOLD_UB_ABS, subprocess_name,
                                                    pop_df=pop_df, sharing_principle=sharing_principle)
        floor_ub = processing.lookup_seuil(dls_df, "Moyenne France", subprocess_name, require_positive=True) if (show_dls and dls_df is not None) else None
        threshold_lower = processing.lookup_threshold(seuils_df, config.THRESHOLD_LOWER_ABS, subprocess_name,
                                                        pop_df=pop_df, sharing_principle=sharing_principle)

        if not show_thresholds:
            # Masque tous les traits de seuil (Lower Safe Bound/Safe Limit/
            # Upper Safe Bound/DLS) : les nuller ici désactive en cascade toute
            # la logique en aval qui en dépend (axhline, rupture d'axe,
            # réservation de place dans bottom_max/ylim). Piloté indépendamment
            # de show_values (valeurs au-dessus des barres).
            threshold_lb = None
            threshold_ub = None
            threshold_lower = None
            floor_ub = None
        elif not display_bounds:
            # Ne masque que les 2 "bounds" (Lower Safe Bound, Upper Safe Bound) :
            # Safe Limit (et DLS, piloté indépendamment par show_dls) reste affiché.
            threshold_ub = None
            threshold_lower = None

        # Rupture d'axe si un seuil (bas et/ou haut) dépasse largement le max observé,
        # pour ne pas écraser les barres en zoomant sur un seuil très lointain.
        use_break_lb = threshold_lb is not None and max_total > 0 and threshold_lb > max_total * 1.5
        use_break_ub = threshold_ub is not None and max_total > 0 and threshold_ub > max_total * 1.6
        use_break = use_break_lb or use_break_ub

        ax_top = None
        # Fraction de la hauteur d'origine du subplot occupée par l'axe
        # principal (ax) une fois rétréci pour laisser la place à ax_top en
        # cas de rupture d'axe : 1.0 si pas de rupture (ax garde toute la
        # hauteur). Nécessaire pour aligner le 0 % sur la même position
        # d'écran d'une ligne à l'autre : un ax rétréci doit viser une
        # fraction de données plus grande pour atteindre la même position
        # physique que dans un subplot non rétréci de même hauteur nominale.
        height_frac = 1.0
        if use_break:
            ax_pos = ax.get_position()
            gap_frac, top_frac = 0.03, 0.1
            gap = ax_pos.height * gap_frac
            top_height = ax_pos.height * top_frac
            bottom_height = ax_pos.height - top_height - gap
            if bottom_height <= 0:
                bottom_height = ax_pos.height * 0.7
                top_height = ax_pos.height * 0.2
                gap = ax_pos.height - bottom_height - top_height
            height_frac = bottom_height / ax_pos.height

            ax.set_position([ax_pos.x0, ax_pos.y0, ax_pos.width, bottom_height])
            ax_top = fig.add_axes([ax_pos.x0, ax_pos.y0 + bottom_height + gap, ax_pos.width, top_height], sharex=ax)
            ax_top.set_facecolor("white")
            ax_top.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax_top.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        # DEUXIÈME PASSE : barres empilées en valeurs absolues (ou, en mode
        # show_variation_bars, en variation par rapport à la référence pour les
        # scénarios non-référence, empilée au-dessus/en-dessous du total de
        # la référence selon le signe de chaque segment).
        baseline_total = total_values[reference_idx]
        baseline_scenario_data = all_scenarios_data[reference_idx]

        # L'axe Y affiche des % relatifs au total du scénario de référence
        # (base year) plutôt que des valeurs absolues : 0% au niveau de la
        # ligne pointillée de base_total, -y% en dessous, +y% au dessus. Les
        # données (barres, seuils, PBA) restent positionnées en valeurs
        # absolues -- seul le texte des graduations est reformaté.
        def _relative_pct(y, _pos, _baseline=baseline_total):
            if not _baseline:
                return f"{y:.0f}"
            pct = (y - _baseline) / _baseline * 100
            return "0%" if abs(pct) < 0.5 else f"{pct:+.0f}%"

        ax.yaxis.set_major_formatter(FuncFormatter(_relative_pct))
        if ax_top is not None:
            ax_top.yaxis.set_major_formatter(FuncFormatter(_relative_pct))

        def value_for(scenario_idx, key, category):
            raw = heights_for(all_scenarios_data[scenario_idx], key, category)
            if show_variation_bars and scenario_idx != reference_idx:
                return raw - heights_for(baseline_scenario_data, key, category)
            return raw

        def draw_segment(category, cat_idx, key, pos_bottom, neg_bottom, hatch=None):
            color_dark = colors.get_dark_shade(colors.get_category_color(category, cat_idx))
            label_suffix = "Import" if hatch else "Domestic"
            values = np.array([value_for(i, key, category) for i in range(n_scenarios)])
            bars_bottom = np.where(values >= 0, pos_bottom, neg_bottom)
            ax.bar(x_pos, values, bar_width, label=f"{category} - {label_suffix}",
                   bottom=bars_bottom, color=color_dark, edgecolor="white", linewidth=0.5, hatch=hatch)
            pos_bottom = pos_bottom + np.where(values >= 0, values, 0)
            neg_bottom = neg_bottom + np.where(values < 0, values, 0)
            return pos_bottom, neg_bottom

        pos_bottom = np.full(n_scenarios, baseline_total if show_variation_bars else 0.0)
        neg_bottom = np.full(n_scenarios, baseline_total if show_variation_bars else 0.0)
        if show_variation_bars:
            pos_bottom[reference_idx] = 0.0
            neg_bottom[reference_idx] = 0.0

        if group_by_category:
            for cat_idx, category in enumerate(categories):
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, "domestique", pos_bottom, neg_bottom)
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, "importé", pos_bottom, neg_bottom, hatch="/")
        else:
            for cat_idx, category in enumerate(categories):
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, "domestique", pos_bottom, neg_bottom)
            for cat_idx, category in enumerate(categories):
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, "importé", pos_bottom, neg_bottom, hatch="/")

        # baseline_total <= max_total par construction (le total de la référence
        # fait partie de total_values), donc toujours dans les limites de ax --
        # pas besoin de la logique de rupture d'axe ici. Sert de repère "0%"
        # pour l'axe relatif, dans tous les modes (pas seulement variation bars).
        ax.axhline(y=baseline_total, color="black", linestyle=":", linewidth=1.3, zorder=9)

        if threshold_lower is not None:
            # Toujours < threshold_lb : jamais concerné par la rupture d'axe
            # (réservée aux seuils lointains, cf. use_break_lb/use_break_ub).
            ax.axhline(y=threshold_lower, color=config.LOWER_SAFE_BOUND_COLOR, linestyle="-", linewidth=2.5,
                       label="Lower Safe Bound", zorder=10)
        if threshold_lb is not None:
            (ax_top if (use_break_lb and ax_top is not None) else ax).axhline(
                y=threshold_lb, color="#0d9a33", linestyle="-", linewidth=2.5, label="Safe Limit", zorder=10)
        if threshold_ub is not None:
            (ax_top if (use_break_ub and ax_top is not None) else ax).axhline(
                y=threshold_ub, color="#bc270a", linestyle="-", linewidth=2.5, label="Upper Safe Bound", zorder=10)
        if floor_ub is not None:
            ax.axhline(y=floor_ub, color="cyan", linestyle="--", linewidth=2.5, label="DLS", zorder=10)

        bottom_max = max(max_total, floor_ub or 0, max_pba)
        if show_variation_bars:
            bottom_max = max(bottom_max, float(np.max(pos_bottom)))
        if threshold_lb is not None and not use_break_lb:
            bottom_max = max(bottom_max, threshold_lb)
        if threshold_ub is not None and not use_break_ub:
            bottom_max = max(bottom_max, threshold_ub)
        # Grille Y commune à tous les subplots : une ligne de grille tous les
        # 10 % du total de référence (baseline_total), mais un chiffre affiché
        # seulement tous les 20 %, de sorte que 0 % et -100 % (= 0 en absolu,
        # borne basse de l'axe) soient toujours visibles, et que l'espacement
        # soit visuellement identique d'un LP à l'autre malgré des unités/
        # échelles absolues différentes. Ne s'applique qu'à l'axe principal :
        # ax_top (petit segment au-dessus d'une rupture d'axe) garde un
        # graduage libre (voir plus bas), les seuils lointains qu'il affiche
        # n'étant pas sur la même grille de 10/20 %.
        if baseline_total:
            step = baseline_total / 10
            ax.yaxis.set_major_locator(MultipleLocator(step * 2))
            ax.yaxis.set_minor_locator(MultipleLocator(step))

        if use_break and ax_top is not None:
            far_thresholds = [t for t, broken in ((threshold_lb, use_break_lb), (threshold_ub, use_break_ub)) if broken]
            top_min, top_max = min(far_thresholds) * 0.98, max(far_thresholds) * 1.02
            if top_min == top_max:
                top_min, top_max = top_min * 0.97, top_max * 1.03
            ax_top.set_ylim(top_min, top_max)

            d = 0.015
            kwargs = dict(transform=ax.transAxes, color="k", clip_on=False, linewidth=1)
            ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)
            ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
            kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False, linewidth=1)
            ax_top.plot((-d, +d), (-d, +d), **kwargs)
            ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

        (ax_top if (use_break and ax_top is not None) else ax).set_title(
            subprocess_name, fontsize=14, fontweight="bold", pad=6 if use_break else 8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_names, rotation=0, ha="center", fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", which="major", alpha=0.3, linestyle="--")
        ax.grid(axis="y", which="minor", alpha=0.15, linestyle="--")

        row_items.append(dict(ax=ax, ax_top=ax_top, use_break=use_break, bottom_max=bottom_max,
                               baseline_total=baseline_total, height_frac=height_frac,
                               pba_values=pba_values, total_values=total_values))

        is_last_in_row = col == 2 or lp_idx == n_lp - 1
        if is_last_in_row:
            # Aligne la graduation 0 % (= baseline_total, en absolu) sur la
            # même position d'écran pour tous les subplots de la ligne, y
            # compris ceux avec une rupture d'axe (dont l'ax principal est
            # rétréci, donc plus court à l'écran qu'un ax non rétréci pour un
            # même ratio baseline/top). On raisonne en "baseline effective" =
            # baseline_total * height_frac : le ratio r = baseline_effective /
            # top commun le plus contraignant est cherché sur cette quantité,
            # puis top = baseline_effective / r pour chaque subplot.
            positive = [it for it in row_items if it["baseline_total"] > 0]
            ratio = min(
                (it["baseline_total"] * it["height_frac"]) / ((it["bottom_max"] or 1.0) * 1.1)
                for it in positive
            ) if positive else None
            for it in row_items:
                own_top = (it["bottom_max"] or 1.0) * 1.1
                effective_baseline = it["baseline_total"] * it["height_frac"]
                top = max(effective_baseline / ratio, own_top) if (ratio and it["baseline_total"] > 0) else own_top
                _finalize_subplot(it["ax"], it["ax_top"], it["use_break"], top,
                                   it["pba_values"], it["total_values"])
            row_items = []

    for idx in range(n_lp, len(positions)):
        row, col = positions[idx]
        fig.add_subplot(gs[row, col]).set_visible(False)

    legend_handles, legend_labels = _build_category_legend(colors.sort_categories(list(legend_categories)))

    if show_thresholds:
        if display_bounds:
            legend_handles.append(Line2D([0], [0], color=config.LOWER_SAFE_BOUND_COLOR, linewidth=3))
            legend_labels.append("Lower Safe Bound")
        legend_handles.append(Line2D([0], [0], color="green", linewidth=3))
        legend_labels.append("Safe Limit")
        if display_bounds:
            legend_handles.append(Line2D([0], [0], color="red", linewidth=3))
            legend_labels.append("Upper Safe Bound")
    if show_pba:
        legend_handles.append(Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=11))
        legend_labels.append("Production-based")
    if show_dls and dls_df is not None and show_thresholds:
        legend_handles.append(Line2D([0], [0], color="cyan", linestyle="--", linewidth=3))
        legend_labels.append("DLS")
    legend_handles.append(Line2D([0], [0], color="black", linestyle=":", linewidth=1.3))
    legend_labels.append("Base year total (0%)")
    if show_variation_bars:
        legend_handles.append(Line2D([0], [0], marker="P", color="black", linestyle="None", markersize=11))
        legend_labels.append("Total footprint")

    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=5, fontsize=16,
               bbox_to_anchor=(0.5, 0.015), frameon=True, fancybox=True, shadow=True)

    fig.suptitle(
        "Planetary boundaries assessment of French NZE Scenarios",
        fontsize=16, fontweight="bold", y=0.98,
    )

    return fig, axes_list
