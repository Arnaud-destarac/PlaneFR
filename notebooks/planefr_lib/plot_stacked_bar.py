"""
Figures of the footprint_decomposition notebook:
  - create_footprint_decomposition_by_scenario : stacked bars in % by indicator,
                                                 for one scenario
  - create_footprint_decomposition             : absolute values, 3x3 grid by LP,
                                                 with boundaries (LP/DLS) and
                                                 axis break if needed

Each one writes the data it plots into an Excel workbook if excel_path is
given (see figure_export).

Only used by the figures in absolute values (multi-scenario Synthesis): the
boundary constants used are therefore fixed to the "_ABS" variant of
planefr_lib.config rather than passed as parameters.
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MultipleLocator

from . import colors, config, figure_export, io, processing
from .processing import DOMESTIC, IMPORTED

CATEGORY_LEGEND_LABELS = figure_export.CATEGORY_LABELS

# sharing_principle="EPC": which UN column of budget_shares.xlsx to use.
#   "region": like plot_overshoot.create_multi_regions_synthesis
#             ("UN – EPC 2019", "UN – EPC 2019-2100" for GHG emissions, and
#             GHG world budgets read in the "GHG emissions (2019)" column).
#   "radial": like plot_radial_synthesis.create_radial_synthesis
#             ("UN – EPC 2050", "UN – EPC 2019-2100" for GHG emissions).
EPC_REFERENCES = ("region", "radial")


def _check_epc_columns(shares_df, epc_reference):
    """Check that budget_shares.xlsx contains the "UN – EPC" columns needed by
    epc_reference (see EPC_REFERENCES)."""
    if epc_reference == "region":
        patterns = {config.REGION_FIGURE_EPC_REF_PATTERN,
                    *config.REGION_FIGURE_EPC_REF_PATTERN_BY_INDICATOR.values()}
    elif epc_reference == "radial":
        patterns = {config.RADIAL_EPC_PATTERN, *config.RADIAL_EPC_PATTERN_BY_INDICATOR.values()}
    else:
        raise ValueError(f"Unknown epc_reference: {epc_reference!r} (expected: {', '.join(EPC_REFERENCES)})")
    for pattern in patterns:
        if not any(re.search(pattern, str(c), flags=re.IGNORECASE) for c in shares_df.columns):
            raise ValueError(
                f"No column of shares_df matches {pattern!r}: regenerate "
                f"budget_shares.xlsx (\"UN – EPC\" columns)."
            )


def _share_boundary(boundaries_df, shares_df, indicator_name, boundary_kind, sharing_principle,
                     epc_reference, use_2019_boundaries=False):
    """France budget (absolute values) = world budget x share of the
    config.RADIAL_EPC_REGION_CODE row ("FR") of budget_shares.xlsx:
      - "EPC": UN EPC column chosen by epc_reference;
      - "CTR": "CTR share ref. ..." column.

    use_2019_boundaries (epc_reference="region" only): True to read the world
    budgets in the 2019 column of boundaries_values.xlsx when it exists (e.g. GHG
    emissions -> "GHG emissions (2019)", see config.region_figure_boundary_column);
    False to keep the column of the indicator ("GHG emissions").
    """
    if epc_reference == "region":
        boundary_column = (config.region_figure_boundary_column(indicator_name, boundaries_df)
                            if use_2019_boundaries else indicator_name)
        column_pattern = config.region_figure_epc_ref_pattern(indicator_name)
    else:
        boundary_column = indicator_name
        column_pattern = config.radial_epc_pattern(indicator_name)
    if sharing_principle != "EPC":
        column_pattern = None  # reference share of the principle (e.g. "CTR share ref.")
    return processing.compute_region_budget(
        boundaries_df, shares_df, boundary_column, config.RADIAL_EPC_REGION_CODE, sharing_principle,
        boundary_kind=boundary_kind, column_pattern=column_pattern,
    )


def _prepare_shares(sharing_principle, shares_df, epc_reference):
    """shares_df to use for a share-based principle ("EPC"/"CTR"), loaded if
    None and checked; None for the other principles."""
    if sharing_principle not in config.BUDGET_SHARE_COLUMN_PATTERNS:
        return None
    if shares_df is None:
        shares_df = io.load_budget_shares_df()
    if sharing_principle == "EPC":
        _check_epc_columns(shares_df, epc_reference)
    return shares_df


def _build_category_legend(categories):
    """Legend shared by the synthesis figures: one square (dark shade) per
    consumption category, followed by a single hatched "Import" pattern.
    F_Y (direct emissions not broken down by the bridge) is shown as
    "Not allocated".
    """
    handles, labels = [], []
    for cat_idx, category in enumerate(categories):
        color_dark = colors.get_dark_shade(colors.get_category_color(category, cat_idx))
        handles.append(Patch(facecolor=color_dark, edgecolor="white"))
        labels.append(CATEGORY_LEGEND_LABELS.get(category, category))

    handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=1))
    labels.append("Import")
    return handles, labels


def _boundaries_note(sharing_principle, epc_reference):
    """"About" text of the workbooks: how the France boundaries are obtained."""
    if sharing_principle is None:
        return "Read as is in boundaries_values.xlsx (France rows)."
    if sharing_principle == "EPC":
        columns = ("UN – EPC 2019, UN – EPC 2019-2100 for GHG emissions" if epc_reference == "region"
                   else "UN – EPC 2050, UN – EPC 2019-2100 for GHG emissions")
        return f"World budget (boundaries_values.xlsx) x share of France, EPC principle ({columns}, budget_shares.xlsx)."
    return (f"World budget (boundaries_values.xlsx) x share of France, {sharing_principle} principle "
            f"(reference share, budget_shares.xlsx).")


def _safe_boundary_getter(boundaries_df, shares_df, sharing_principle, epc_reference, use_2019_boundaries):
    """indicator_name -> lower boundary of France (Safe Boundary) of the stacked
    bars: share-based (shares_df given) or read as is in boundaries_df."""
    def safe_boundary_of(indicator_name):
        if shares_df is not None:
            return _share_boundary(boundaries_df, shares_df, indicator_name, "lb", sharing_principle,
                                    epc_reference, use_2019_boundaries=use_2019_boundaries)
        return processing.lookup_boundary(boundaries_df, config.BOUNDARY_LB_ABS, indicator_name)
    return safe_boundary_of


def _by_scenario_tables(scenario_name, data_by_indicator, boundaries_df, reference_data, safe_boundary_of):
    """Rows of the "Footprint" and "Indicators" sheets of
    create_footprint_decomposition_by_scenario for one scenario."""
    indicator_rows = []
    for indicator_name, data in data_by_indicator.items():
        if data is None:
            continue
        total = processing.extract_value(data, "cba")
        ref_total = (processing.extract_value(reference_data.get(indicator_name), "cba")
                     if reference_data is not None else None)
        safe_boundary = figure_export.clean_value(safe_boundary_of(indicator_name))
        indicator_rows.append({
            "Scenario": scenario_name,
            "Indicator": indicator_name,
            "Unit": figure_export.indicator_unit(boundaries_df, indicator_name),
            "Total footprint (CBA)": total,
            "Variation vs reference scenario (%)": figure_export.variation_pct(total, ref_total),
            "Safe Boundary (France)": safe_boundary,
            "Overshoot of the Safe Boundary (x)": figure_export.ratio(total, safe_boundary),
        })
    footprint = figure_export.footprint_rows(scenario_name, data_by_indicator, boundaries_df)
    return footprint, indicator_rows


def create_footprint_decomposition_by_scenario(data_by_indicator, boundaries_df, indicator_to_lp,
                              scenario_name="", reference_data=None, ax=None, show_legend=True,
                              group_by_category=True, sharing_principle=None,
                              shares_df=None, epc_reference="region", use_2019_boundaries=None,
                              show_values=True, excel_path=None):
    """Stacked bars in % showing the LPs by indicator: dom (solid) + imp
    (hatched) stacked by consumption category. Annotates the absolute total,
    the variation relative to the reference scenario (if given), and the
    overshoot of the lower boundary.

    Args:
        data_by_indicator: {indicator_name: result of process_single_indicator_scenario}
        indicator_to_lp: indicator -> [LP] mapping (not used directly here,
            kept in the signature for consistency with the callers)
        reference_data: data_by_indicator of a reference scenario, to display
            the relative variation (omit for the reference scenario itself)
        ax: existing matplotlib axis; if None, a new figure is created
        group_by_category: True (default) to stack dom/imp one after the other
            for each category (cat1-dom, cat1-imp, cat2-dom, cat2-imp, ...);
            False to group by origin (all the dom categories, then all the imp
            categories)
        sharing_principle: how the lower boundary (LB) of France is obtained:
            - None (default): read as is in boundaries_df ("Equality" row);
            - "EPC": world budget ("Safe Boundary") x UN EPC share of the "FR" row
              of budget_shares.xlsx, column chosen by epc_reference;
            - "CTR": world budget x "CTR share ref." share of the "FR" row.
        shares_df: budget_shares.xlsx (io.load_budget_shares_df) -- used for
            "EPC"/"CTR"; loaded automatically if None.
        epc_reference: "region" (default) = same limits as
            create_multi_regions_synthesis ("UN – EPC 2019",
            except GHG emissions: "UN – EPC 2019-2100" on the "GHG emissions
            (2019)" budget); "radial" = same limits as
            create_radial_synthesis ("UN – EPC 2050", except GHG
            emissions: "UN – EPC 2019-2100").
        use_2019_boundaries: with epc_reference="region", True to take the
            world budgets of "GHG emissions (2019)" instead of "GHG emissions"
            (like the by-region figure, which only shows 2019 scenarios);
            False to keep "GHG emissions". None (default): True if
            scenario_name contains "2019" (e.g. "2019_FR"), False otherwise.
        show_values: True (default) to annotate the absolute total above each
            bar and the overshoot level of the lower boundary just below the
            top of the bar; False hides these two annotations (and the
            "Overshoot" legend entry). The variation relative to the reference
            scenario remains displayed.
        excel_path: if given, Excel workbook of the plotted data (see
            figure_export): "Footprint" sheet (value and share of each
            category x origin segment) and "Indicators" sheet (total,
            variation vs reference, Safe Boundary, overshoot).

    Returns:
        (fig, ax) if ax=None was given as input, otherwise (None, ax).
    """
    if use_2019_boundaries is None:
        use_2019_boundaries = "2019" in str(scenario_name)
    shares_df = _prepare_shares(sharing_principle, shares_df, epc_reference)
    is_share = shares_df is not None

    safe_boundary_of = _safe_boundary_getter(boundaries_df, shares_df, sharing_principle, epc_reference,
                                       use_2019_boundaries)

    indicators = list(data_by_indicator.keys())
    n_indicators = len(indicators)

    if not any(data is not None for data in data_by_indicator.values()):
        print("No valid data to plot")
        return (None, None) if ax is None else (None, ax)

    if excel_path is not None:
        footprint, indicator_rows = _by_scenario_tables(scenario_name, data_by_indicator, boundaries_df,
                                                        reference_data, safe_boundary_of)
        figure_export.write_figure_excel(excel_path, {"Footprint": footprint, "Indicators": indicator_rows}, about={
            "Figure": f"Footprint decomposition - {scenario_name} (consumption-based)",
            "Generated by": "planefr_lib.plot_stacked_bar.create_footprint_decomposition_by_scenario",
            "Content": "Consumption-based footprint of each indicator, broken down by consumption category "
                       "and origin (bars in % of the total); total, variation vs the reference scenario and "
                       "overshoot of the Safe Boundary.",
            "Unit": "Unit column (\"Figures unit\" of boundaries_values.xlsx)",
            "Boundaries": _boundaries_note(sharing_principle, epc_reference),
        })

    all_categories = set()
    for data in data_by_indicator.values():
        if data is not None:
            all_categories.update(data["categories"])
    categories = colors.sort_categories(list(all_categories))

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))
    else:
        fig = None

    def heights_for(key, category):
        return np.array([
            data_by_indicator[sp][key].get(category, 0)
            if (data_by_indicator[sp] is not None and key in data_by_indicator[sp]) else 0
            for sp in indicators
        ])

    bar_width = 0.6
    x_pos = np.arange(n_indicators)

    total_values = np.zeros(n_indicators)
    for category in categories:
        total_values += heights_for(DOMESTIC, category) + heights_for(IMPORTED, category)

    def draw_segment(category, cat_idx, key, bottom_heights, hatch=None):
        color_dark = colors.get_dark_shade(colors.get_category_color(category, cat_idx))
        label_suffix = "Import" if hatch else "Domestic"
        heights_pct = np.where(total_values > 0, heights_for(key, category) / total_values * 100, 0)
        ax.bar(x_pos, heights_pct, bar_width, label=f"{category} - {label_suffix}",
               bottom=bottom_heights, color=color_dark, edgecolor="white", linewidth=0.5, hatch=hatch)
        return heights_pct

    bottom_heights = np.zeros(n_indicators)
    if group_by_category:
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, DOMESTIC, bottom_heights)
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, IMPORTED, bottom_heights, hatch="/")
    else:
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, DOMESTIC, bottom_heights)
        for cat_idx, category in enumerate(categories):
            bottom_heights = bottom_heights + draw_segment(category, cat_idx, IMPORTED, bottom_heights, hatch="/")

    for idx, total_val in enumerate(total_values):
        sp = indicators[idx]

        if show_values:
            unit_val = processing.lookup_first_text(boundaries_df, config.UNIT_ROW_ABS, sp)
            ax.text(idx, 102, f"{total_val:.0f}{f' {unit_val}' if unit_val else ''}",
                    ha="center", va="bottom", fontweight="bold", fontsize=10.5)

        if reference_data is not None and sp in reference_data:
            ref_total = reference_data[sp][DOMESTIC].sum() + reference_data[sp][IMPORTED].sum()
            if ref_total > 0:
                variation_pct = (total_val - ref_total) / ref_total * 100
                text_var = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
                ax.text(idx, 97, text_var, ha="center", va="bottom", fontweight="bold", fontsize=11, color="black")

        if not show_values:
            continue
        boundary = safe_boundary_of(sp)
        if boundary:
            overshoot_pct = total_val / boundary
            color = "red" if overshoot_pct > 1 else "lightgreen"
            y_pos = 95 if reference_data is not None else 98
            ax.text(idx, y_pos, f"{overshoot_pct:.1f}", ha="center", va="top",
                    fontweight="bold", fontsize=14, color=color)

    ax.set_ylabel("Footprint share (%)", fontsize=12, fontweight="bold")
    if scenario_name:
        ax.set_title(f"{scenario_name} - Consumption-based", fontsize=12, fontweight="bold", pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(indicators, rotation=30, ha="center", fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if show_legend:
        handles, labels = _build_category_legend(categories)
        if show_values:
            handles.append(Line2D([0], [0], marker="$2.5$", color="red", linestyle="None", markersize=13))
            labels.append("Overshoot")
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=13)

    if fig is not None:
        plt.tight_layout()

    return (fig, ax) if fig is not None else (None, ax)


def create_footprint_decomposition(all_scenarios_data, boundaries_df, indicator_to_lp_list, scenario_names,
                                   dls_df=None, group_by_category=True, show_pba=True, show_dls=True,
                                   show_values=True, display_bounds=True,
                                   show_variation_bars=False,
                                   sharing_principle=None, scenario_group=None,
                                   shares_df=None, epc_reference="radial", excel_path=None):
    """Alternative synthesis figure: 7 LPs in a 3x3 grid (1 bar per scenario,
    1 subplot per LP), absolute values (not normalized). Displays the
    environmental ceiling (Safe Boundary, High-Risk Boundary) and the social floor
    (DLS) if they exist, with an automatic axis break when the upper ceiling
    is far above the observed maximum.

    Args:
        group_by_category: see create_footprint_decomposition_by_scenario
        show_pba: True (default) to display the production-based (PBA) cross
            and its annotations on each bar; False to hide them entirely
            (bars, text, legend).
        show_dls: True (default) to display the social floor (DLS), as well as
            DLS lower bound and DLS upper bound (shades of blue, light ->
            dark), when dls_df provides a value for the LP; False to ignore it
            even if dls_df is given. National values read as is in the DLS
            sheet (processing.lookup_dls): "DLS ..." rows for the base-year
            scenarios (config.is_base_year_scenario: 2015, 2019,
            Base_year_2015, 2019_FR...), "DLS 2050 ..." for the others.
        scenario_group: displayed scenario group (config.get_scenario_folders),
            only used to choose the DLS rows (see show_dls).
        show_values: True (default) to display the values/variations above
            the bars (CBA and PBA); False hides them (the bars, the PBA cross
            and the boundary lines are unchanged).
        display_bounds: True (default) displays Lower Safe Boundary and High-Risk Boundary in addition to Safe
            Limit (always displayed). False hides them (only Safe Boundary -- and
            DLS if show_dls -- remains displayed).
        show_variation_bars: False (default) for the usual display (each bar =
            absolute footprint stacked by category). True to display, for all
            the scenarios except the reference, only the variation with respect
            to the reference: each category/origin segment (domestic or
            imported) is stacked above the reference total (the "0 ordinate"
            of the view) if it increases, or stacked below if it decreases,
            with a height equal to the absolute value of the variation. The
            reference bar remains unchanged (stacked from 0) and serves as a
            visual marker for this zero level (dotted line). A cross marks, for
            each scenario, the actual total of its footprint (useful when some
            segments go up and others go down, to see where the net balance is).
        sharing_principle: see create_footprint_decomposition_by_scenario -- applies here to
            boundary_lower, boundary_lb AND boundary_ub.
        shares_df, epc_reference: see create_footprint_decomposition_by_scenario (used if
            sharing_principle="EPC"/"CTR"). Here epc_reference="radial" by default:
            same limits as create_radial_synthesis ("UN – EPC 2050",
            except GHG emissions: "UN – EPC 2019-2100").
        excel_path: if given, Excel workbook of the plotted data (see
            figure_export): "Footprint" (category x origin segments, absolute
            values), "Indicators" (CBA and PBA totals and their variation vs
            the reference scenario) and "Boundaries" (Lower Safe Boundary, Safe
            Limit, High-Risk Boundary and DLS of France, by indicator) sheets.
            The values are always absolute, even with show_variation_bars.

    Returns:
        (fig, axes_list)
    """
    shares_df = _prepare_shares(sharing_principle, shares_df, epc_reference)
    boundary_rows = []  # "Boundaries" sheet of excel_path

    def boundary(boundary_row, boundary_kind, indicator_name):
        if shares_df is not None:
            return _share_boundary(boundaries_df, shares_df, indicator_name, boundary_kind,
                                    sharing_principle, epc_reference)
        return processing.lookup_boundary(boundaries_df, boundary_row, indicator_name)

    reference_idx = config.REFERENCE_SCENARIO_IDX
    # The reference scenario (base_year) keeps its position, followed by the
    # trend scenario, then the other scenarios by decreasing mean variation of
    # the footprint vs the reference (smallest decrease / largest increase next
    # to the trend scenario, largest decrease the furthest away) -- same logic
    # as create_radial_synthesis (processing.sort_scenarios_by_reduction).
    value_keys = ["cba"] + (["pba"] if show_pba else [])
    display_order = processing.sort_scenarios_by_reduction(all_scenarios_data, scenario_names, reference_idx, value_keys)
    scenario_names = [scenario_names[i] for i in display_order]
    all_scenarios_data = [all_scenarios_data[i] for i in display_order]
    indicator_to_lp_list = [indicator_to_lp_list[i] for i in display_order]

    first_scenario_data = all_scenarios_data[reference_idx]
    indicators = list(first_scenario_data.keys())
    n_lp = len(indicators)

    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.2, wspace=0.15, top=0.93, bottom=0.12, left=0.045, right=0.99)
    positions = [(r, c) for r in range(3) for c in range(3)]

    reference_idx = config.REFERENCE_SCENARIO_IDX
    axes_list = []
    legend_categories = set()
    # DLS levels: (key, row of the DLS sheet, color, label, scope).
    # Chosen per scenario, whatever the group (config.is_base_year_scenario):
    # "DLS ..." for the base-year scenarios (2015, 2019, Base_year_2015,
    # 2019_FR...), "DLS 2050 ..." for the others. All of one kind: the 3 levels
    # on all the bars; mixed: base-year DLS on the base-year bars ("base"),
    # DLS 2050 on the others ("others"), like the radial synthesis.
    base_idx = {i for i, name in enumerate(scenario_names) if config.is_base_year_scenario(name)}
    if len(base_idx) in (0, len(scenario_names)):
        prefix, label_prefix = (("DLS", "DLS Base year") if base_idx else ("DLS 2050", "DLS 2050"))
        dls_levels = [(key, prefix + suffix, color, label_prefix + label[len("DLS"):], "all")
                      for key, suffix, color, label in config.DLS_LEVELS]
    else:
        dls_levels = [("dls_base", "DLS", config.DLS_COLOR, "DLS Base year", "base"),
                      ("dls_2050", "DLS 2050", config.DLS_2050_COLOR, "DLS 2050", "others")]
    dls_levels_found = {key: False for key, *_ in dls_levels}  # DLS levels drawn (legend)

    def _finalize_subplot(ax, ax_top, use_break, top, pba_values, total_values):
        """Second pass (after aligning 0 % across the row): sets the upper
        limit of the axis, then draws everything that depends on ax.get_ylim()
        (PBA cross, total marker in variation mode, value texts).
        """
        ax.set_ylim(0, top)

        # PBA (production-based accounting): bold black cross on each bar, to
        # compare visually with the top of the bar (CBA, consumption-based).
        pba_axes = [None] * len(pba_values)
        for scenario_idx, pba_val in enumerate(pba_values):
            if pba_val is None or not np.isfinite(pba_val):
                continue
            target_ax = ax_top if (use_break and ax_top is not None and pba_val > ax.get_ylim()[1]) else ax
            pba_axes[scenario_idx] = target_ax
            target_ax.scatter(scenario_idx, pba_val, marker="X", s=140, color="black",
                               linewidths=1.5, zorder=12)

        # Variation mode: "+" cross marking, for each scenario, the actual total
        # of its footprint (CBA) -- useful because the net balance of a scenario
        # does not necessarily match the top of the segments stacked upwards
        # when some segments go up and others go down.
        if show_variation_bars:
            for scenario_idx, total_val in enumerate(total_values):
                target_ax = ax_top if (use_break and ax_top is not None and total_val > ax.get_ylim()[1]) else ax
                target_ax.scatter(scenario_idx, total_val, marker="P", s=170, color="black",
                                   edgecolors="white", linewidths=1, zorder=14)

        if show_values:
            ref_total = total_values[reference_idx]
            ref_pba = pba_values[reference_idx]
            ax_margin = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05

            for scenario_idx, total_val in enumerate(total_values):
                pba_val = pba_values[scenario_idx]
                pba_ax = pba_axes[scenario_idx]
                cba_y, cba_va = total_val + ax_margin, "bottom"

                if scenario_idx == reference_idx:
                    ax.text(scenario_idx, cba_y, f"{total_val:.0f}",
                            ha="center", va=cba_va, fontweight="bold", fontsize=15, color="black", zorder=13)
                elif ref_total > 0:
                    variation_pct = (total_val - ref_total) / ref_total * 100
                    text_var = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
                    ax.text(scenario_idx, cba_y, text_var,
                            ha="center", va=cba_va, fontweight="bold", fontsize=15, color="black", zorder=13)

                # PBA text (variation relative to the reference only, no total
                # for the reference), below the cross.
                if pba_val is None or not np.isfinite(pba_val) or pba_ax is None:
                    continue
                pba_margin = (pba_ax.get_ylim()[1] - pba_ax.get_ylim()[0]) * 0.05
                pba_y = pba_val - pba_margin

                if scenario_idx == reference_idx:
                    pass  # PBA total of the reference not displayed
                elif ref_pba is not None and np.isfinite(ref_pba) and ref_pba > 0:
                    pba_variation_pct = (pba_val - ref_pba) / ref_pba * 100
                    pba_text_var = f"+{pba_variation_pct:.0f}%" if pba_variation_pct > 0 else f"{pba_variation_pct:.0f}%"
                    pba_ax.text(scenario_idx, pba_y, pba_text_var,
                                ha="center", va="top", fontweight="bold", style="italic", fontsize=15,
                                color="black", zorder=13)

    row_items = []

    for lp_idx, indicator_name in enumerate(indicators):
        if lp_idx >= len(positions):
            break

        row, col = positions[lp_idx]
        ax = fig.add_subplot(gs[row, col])
        axes_list.append(ax)

        n_scenarios = len(all_scenarios_data)
        bar_width = 0.6
        x_pos = np.arange(n_scenarios)

        categories = first_scenario_data[indicator_name]["categories"]
        legend_categories.update(categories)

        def heights_for(scenario_data, key, category):
            if indicator_name not in scenario_data:
                return 0
            return scenario_data[indicator_name][key].get(category, 0)

        # FIRST PASS: absolute totals by scenario (to size the Y axis)
        total_values = np.zeros(n_scenarios)
        for category in categories:
            for scenario_idx, scenario_data in enumerate(all_scenarios_data):
                total_values[scenario_idx] += (
                    heights_for(scenario_data, DOMESTIC, category)
                    + heights_for(scenario_data, IMPORTED, category)
                )

        max_total = float(np.max(total_values)) if total_values.size else 0.0

        if show_pba:
            pba_values = [
                scenario_data[indicator_name].get("pba") if indicator_name in scenario_data else None
                for scenario_data in all_scenarios_data
            ]
            finite_pba = [float(v) for v in pba_values if v is not None and np.isfinite(v)]
            max_pba = max(finite_pba) if finite_pba else 0.0
        else:
            pba_values = [None] * n_scenarios
            max_pba = 0.0

        boundary_lb = boundary(config.BOUNDARY_LB_ABS, "lb", indicator_name)
        boundary_ub = boundary(config.BOUNDARY_UB_ABS, "ub", indicator_name)
        # DLS (and its bounds), by key of config.DLS_LEVELS; None if missing for this LP
        dls_values = {key: None for key, *_ in dls_levels}
        if show_dls and dls_df is not None:
            for key, dls_row, *_ in dls_levels:
                dls_values[key] = processing.lookup_dls(dls_df, dls_row, indicator_name)
        boundary_lower = boundary(config.BOUNDARY_LOWER_ABS, "lower", indicator_name)

        # Exported before the display options hide some of them
        boundary_rows.append({
            "Indicator": indicator_name,
            "Unit": figure_export.indicator_unit(boundaries_df, indicator_name),
            "Lower Safe Boundary": figure_export.clean_value(boundary_lower),
            "Safe Boundary": figure_export.clean_value(boundary_lb),
            "High-Risk Boundary": figure_export.clean_value(boundary_ub),
            **{dls_label: figure_export.clean_value(dls_values[key]) for key, _, _, dls_label, _ in dls_levels},
        })

        if not display_bounds:
            # Only hides the 2 "bounds" (Lower Safe Boundary, High-Risk Boundary):
            # Safe Boundary (and DLS, controlled independently by show_dls) remains.
            boundary_ub = None
            boundary_lower = None

        # Axis break if a boundary (lower and/or upper) is far above the
        # observed max, so as not to squash the bars by zooming on a very
        # distant boundary.
        use_break_lb = boundary_lb is not None and max_total > 0 and boundary_lb > max_total * 1.5
        use_break_ub = boundary_ub is not None and max_total > 0 and boundary_ub > max_total * 1.6
        use_break = use_break_lb or use_break_ub

        ax_top = None
        # Fraction of the original height of the subplot taken by the main axis
        # (ax) once shrunk to make room for ax_top in case of an axis break: 1.0
        # without break (ax keeps the full height). Needed to align 0 % on the
        # same screen position from one row to another: a shrunk ax must target
        # a larger data fraction to reach the same physical position as in a
        # non-shrunk subplot of the same nominal height.
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

        # SECOND PASS: stacked bars in absolute values (or, in
        # show_variation_bars mode, as a variation with respect to the
        # reference for the non-reference scenarios, stacked above/below the
        # reference total depending on the sign of each segment).
        baseline_total = total_values[reference_idx]
        baseline_scenario_data = all_scenarios_data[reference_idx]

        # The Y axis displays % relative to the total of the reference scenario
        # (base year) rather than absolute values: 0% at the level of the dotted
        # base_total line, -y% below, +y% above. The data (bars, boundaries,
        # PBA) remain positioned in absolute values -- only the tick text is
        # reformatted.
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
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, DOMESTIC, pos_bottom, neg_bottom)
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, IMPORTED, pos_bottom, neg_bottom, hatch="/")
        else:
            for cat_idx, category in enumerate(categories):
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, DOMESTIC, pos_bottom, neg_bottom)
            for cat_idx, category in enumerate(categories):
                pos_bottom, neg_bottom = draw_segment(category, cat_idx, IMPORTED, pos_bottom, neg_bottom, hatch="/")

        # baseline_total <= max_total by construction (the reference total is
        # part of total_values), hence always within the limits of ax -- no need
        # for the axis break logic here. Serves as the "0%" marker of the
        # relative axis, in all modes (not only variation bars).
        ax.axhline(y=baseline_total, color="black", linestyle=":", linewidth=1.3, zorder=9)

        if boundary_lower is not None:
            # Always < boundary_lb: never concerned by the axis break
            # (reserved for distant boundaries, see use_break_lb/use_break_ub).
            ax.axhline(y=boundary_lower, color=config.LOWER_SAFE_BOUNDARY_COLOR, linestyle="-", linewidth=2.5,
                       label="Lower Safe Boundary", zorder=10)
        if boundary_lb is not None:
            (ax_top if (use_break_lb and ax_top is not None) else ax).axhline(
                y=boundary_lb, color="#0d9a33", linestyle="-", linewidth=2.5, label="Safe Boundary", zorder=10)
        if boundary_ub is not None:
            (ax_top if (use_break_ub and ax_top is not None) else ax).axhline(
                y=boundary_ub, color="#bc270a", linestyle="-", linewidth=2.5, label="High-Risk Boundary", zorder=10)
        for key, _, dls_color, dls_label, scope in dls_levels:
            if dls_values[key] is not None:
                dls_levels_found[key] = True
                if scope == "all":
                    ax.axhline(y=dls_values[key], color=dls_color, linestyle="--", linewidth=2.5,
                               label=dls_label, zorder=10)
                    continue
                # Line limited to the base-year bars ("base") or to the 2050
                # scenarios ("others"): each bar of the set, widened up to
                # the middle of the gaps (to the edge of the axis at both
                # ends). Added via add_artist (and not ax.plot) so as not to
                # change the data limits of the axis: the ±10 bounds are simply
                # clipped at the edge of the axis.
                idxs = [i for i in range(n_scenarios) if (i in base_idx) == (scope == "base")]
                spans = [(-10 if i == 0 else i - 0.5, n_scenarios + 10 if i == n_scenarios - 1 else i + 0.5)
                         for i in idxs]
                for x0, x1 in spans:
                    dls_line = Line2D([x0, x1], [dls_values[key]] * 2, color=dls_color, linestyle="--",
                                      linewidth=2.5, zorder=10, transform=ax.transData)
                    ax.add_artist(dls_line)
                    dls_line.set_clip_path(ax.patch)

        bottom_max = max(max_total, *(v for v in dls_values.values() if v is not None), max_pba)
        if show_variation_bars:
            bottom_max = max(bottom_max, float(np.max(pos_bottom)))
        if boundary_lb is not None and not use_break_lb:
            bottom_max = max(bottom_max, boundary_lb)
        if boundary_ub is not None and not use_break_ub:
            bottom_max = max(bottom_max, boundary_ub)
        # Y grid common to all the subplots: a grid line every 10 % of the
        # reference total (baseline_total), but a label only every 20 %, so that
        # 0 % and -100 % (= 0 in absolute terms, lower bound of the axis) are
        # always visible, and that the spacing is visually identical from one LP
        # to another despite different units/absolute scales. Only applies to
        # the main axis: ax_top (small segment above an axis break) keeps free
        # ticks (see below), the distant boundaries it shows not being on the
        # same 10/20 % grid.
        if baseline_total:
            step = baseline_total / 10
            ax.yaxis.set_major_locator(MultipleLocator(step * 2))
            ax.yaxis.set_minor_locator(MultipleLocator(step))

        if use_break and ax_top is not None:
            far_boundaries = [t for t, broken in ((boundary_lb, use_break_lb), (boundary_ub, use_break_ub)) if broken]
            top_min, top_max = min(far_boundaries) * 0.98, max(far_boundaries) * 1.02
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
            indicator_name, fontsize=14, fontweight="bold", pad=6 if use_break else 8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(scenario_names, rotation=0, ha="center", fontweight="bold", fontsize=10.5)
        ax.grid(axis="y", which="major", alpha=0.3, linestyle="--")
        ax.grid(axis="y", which="minor", alpha=0.15, linestyle="--")

        row_items.append(dict(ax=ax, ax_top=ax_top, use_break=use_break, bottom_max=bottom_max,
                               baseline_total=baseline_total, height_frac=height_frac,
                               pba_values=pba_values, total_values=total_values))

        is_last_in_row = col == 2 or lp_idx == n_lp - 1
        if is_last_in_row:
            # Aligns the 0 % tick (= baseline_total, in absolute terms) on the
            # same screen position for all the subplots of the row, including
            # those with an axis break (whose main ax is shrunk, hence shorter
            # on screen than a non-shrunk ax for the same baseline/top ratio).
            # The reasoning uses the "effective baseline" = baseline_total *
            # height_frac: the most constraining common ratio r =
            # effective_baseline / top is searched on this quantity, then top =
            # effective_baseline / r for each subplot.
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

    if display_bounds:
        legend_handles.append(Line2D([0], [0], color=config.LOWER_SAFE_BOUNDARY_COLOR, linewidth=3))
        legend_labels.append("Lower Safe Boundary")
    legend_handles.append(Line2D([0], [0], color="green", linewidth=3))
    legend_labels.append("Safe Boundary")
    if display_bounds:
        legend_handles.append(Line2D([0], [0], color="red", linewidth=3))
        legend_labels.append("High-Risk Boundary")
    if show_pba:
        legend_handles.append(Line2D([0], [0], marker="X", color="black", linestyle="None", markersize=11))
        legend_labels.append("Production-based")
    if show_dls and dls_df is not None:
        for key, _, dls_color, dls_label, _ in dls_levels:
            if dls_levels_found[key]:
                legend_handles.append(Line2D([0], [0], color=dls_color, linestyle="--", linewidth=3))
                legend_labels.append(dls_label)
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

    if excel_path is not None:
        plotted = indicators[:len(positions)]
        reference_data = all_scenarios_data[reference_idx]
        footprint, indicator_rows = [], []
        for scenario_idx, (scenario_name, scenario_data) in enumerate(zip(scenario_names, all_scenarios_data)):
            footprint += figure_export.footprint_rows(scenario_name, scenario_data, boundaries_df, plotted)
            for indicator_name in plotted:
                payload = scenario_data.get(indicator_name)
                if payload is None:
                    continue
                cba = processing.extract_value(payload, "cba")
                pba = processing.extract_value(payload, "pba")
                # No variation for the reference scenario itself
                ref_payload = None if scenario_idx == reference_idx else reference_data.get(indicator_name)
                indicator_rows.append({
                    "Scenario": scenario_name,
                    "Indicator": indicator_name,
                    "Unit": figure_export.indicator_unit(boundaries_df, indicator_name),
                    "Consumption-based (CBA)": cba,
                    "CBA variation vs reference scenario (%)":
                        figure_export.variation_pct(cba, processing.extract_value(ref_payload, "cba")),
                    "Production-based (PBA)": pba,
                    "PBA variation vs reference scenario (%)":
                        figure_export.variation_pct(pba, processing.extract_value(ref_payload, "pba")),
                })
        figure_export.write_figure_excel(
            excel_path,
            {"Footprint": footprint, "Indicators": indicator_rows, "Boundaries": boundary_rows},
            about={
                "Figure": "Planetary boundaries assessment of French NZE Scenarios (footprint decomposition)",
                "Generated by": "planefr_lib.plot_stacked_bar.create_footprint_decomposition",
                "Content": "Consumption-based footprint of France by indicator and scenario, broken down by "
                           "consumption category and origin; consumption- and production-based totals; "
                           "boundaries of France (planetary boundaries and decent living standards).",
                "Scenario group": scenario_group,
                "Reference scenario": scenario_names[reference_idx],
                "Scenario order": ", ".join(scenario_names),
                "Unit": "Unit column (\"Figures unit\" of boundaries_values.xlsx)",
                "Boundaries": _boundaries_note(sharing_principle, epc_reference),
                "DLS": "Decent living standards, DLS sheet of boundaries_values.xlsx (national values)",
            },
        )

    return fig, axes_list
