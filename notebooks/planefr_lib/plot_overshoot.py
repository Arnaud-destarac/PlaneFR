"""
"Overshoot / Safe Operating Space" figure: 1 row per planetary boundary (LP),
position of each scenario on the x axis relative to the lower limit (1L =
equality boundary), green -> orange -> purple gradient background (safe zone
-> increasing risk -> high-risk zone), axis break if a scenario exceeds
x_main_max times the lower limit.

Public function: create_multi_regions_synthesis (multi-regions_synthesis
notebook) -- absolute values (nothing is divided by the population, neither
the footprints nor the budgets), budget of each country/region obtained by
multiplying the world budget by its share read in budget_shares.xlsx
("EPC"/"CTR" principles). Writes its data into an Excel workbook if
excel_path is given (see figure_export).
"""

import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from . import colors, config, figure_export, processing
from .processing import DOMESTIC, IMPORTED

# Font size of the X axis ticks in "nL" (e.g. 1L, 2L... 7L, 8L...), shared by
# the main axis and the break axis so that they stay identical.
AXIS_TICK_LABEL_FONTSIZE = 13

# Height (fraction of bar height) taken by the single line of the title
# (LP name), at fontsize=13 (about half of the former 2-line block, 0.35).
# Used to size `hspace` (the space between bars) automatically from the two
# adjustable gaps title_gap_below_bar/title_gap_above_bar, so that the title
# always has room to fit.
TITLE_BLOCK_HEIGHT = 0.18

# show_value_overshoot=True only -- height (fraction of bar height) reserved
# above each bar for its own "nL" axis (ticks + labels), added to `hspace` and
# to the title/bar gap so that the title goes above this axis instead of
# overlapping it.
VALUE_AXIS_BLOCK_HEIGHT = 0.30
VALUE_AXIS_TICK_LABEL_FONTSIZE = 11

# ============================================================================
# GENERIC DRAWING/FORMATTING HELPERS
# ============================================================================


def _total_footprint(lp_payload):
    """Total footprint (domestic + imported) for a indicator."""
    return float(lp_payload[DOMESTIC].sum() + lp_payload[IMPORTED].sum())


def _draw_gradient_segment(ax, x0, x1, seg_colors, y0=0.10, y1=0.90):
    """Draw a horizontal segment with a continuous gradient between `seg_colors`."""
    if x1 <= x0:
        return
    grad = np.linspace(0, 1, 512).reshape(1, -1)
    cmap = LinearSegmentedColormap.from_list("lp_grad", seg_colors)
    ax.imshow(grad, extent=[x0, x1, y0, y1], aspect="auto", cmap=cmap, interpolation="bicubic", zorder=0)


def _draw_lp_background(ax, x_end, safe_x=None, risk_x=None):
    """Background safe (green) -> increasing risk (yellow/orange) -> high risk
    (red/purple), with transitions at safe_x (Safe Boundary, SAFE_BOUNDARY_REL by
    default) and risk_x (High-Risk Boundary, RISK_TRANSITION_REL by default).

    The three zones are drawn over their whole extent in data coordinates
    ([0, safe_x], [safe_x, risk_x], [risk_x, x_end]) and clipped by the axes:
    called with the same x_end on the main axis and on the break axis, the
    gradient continues from one to the other (each axis shows its portion).
    The xlim of `ax` must be set beforehand."""
    safe_x = config.SAFE_BOUNDARY_REL if safe_x is None else safe_x
    risk_x = config.RISK_TRANSITION_REL if risk_x is None else max(risk_x, safe_x)
    _draw_gradient_segment(ax, 0, safe_x, ["#0b7d3e", "#73bf44"])
    _draw_gradient_segment(ax, safe_x, risk_x, ["#f5dd3d", "#eb8f1e"])
    _draw_gradient_segment(ax, risk_x, max(x_end, risk_x), HIGH_RISK_COLORS)


HIGH_RISK_COLORS = ["#e7731c", "#d62324", "#8f1f64", "#4f256d"]
# Beyond the break: continuation of the High-Risk gradient, from its end color
# towards a darker purple (further = darker).
HIGH_RISK_BEYOND_BREAK_COLORS = ["#4f256d", "#1f0b2e"]


def _draw_break_background(ax, x_min, x_max, x_main_end, safe_x=None, risk_x=None):
    """Background of the break axis ([x_min, x_max], beyond x_main_end).

    - Increasing Risk: same gradient as on the main axis ([safe_x, risk_x],
      clipped), if the UB lies beyond the break.
    - High-Risk: if the UB is before the break (the main axis already ends on
      the end color of the High-Risk gradient), the gradient continues from
      that color towards a darker purple; if the UB is beyond the break, the
      full High-Risk gradient starts at the UB.
    The main axis is unchanged, so that a given nL keeps the same color on
    every row."""
    if x_max <= x_min:
        return
    safe_x = config.SAFE_BOUNDARY_REL if safe_x is None else safe_x
    risk_x = config.RISK_TRANSITION_REL if risk_x is None else max(risk_x, safe_x)
    if risk_x > x_min:
        _draw_gradient_segment(ax, safe_x, risk_x, ["#f5dd3d", "#eb8f1e"])
    if risk_x >= x_max:
        return
    if risk_x > x_main_end:
        _draw_gradient_segment(ax, risk_x, x_max, HIGH_RISK_COLORS)
    else:
        _draw_gradient_segment(ax, x_min, x_max, HIGH_RISK_BEYOND_BREAK_COLORS)


def _set_break_axis_ticks(ax_ext):
    """Readable integer ticks on the right segment after the axis break."""
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


def _regular_rel_ticks(rel_top, max_ticks=7):
    """Ticks of an "nL" axis specific to a bar (show_value_overshoot): 1L, then
    the multiples of a "round" step (1, 2, 2.5, 5 x 10^k) between 1L
    (excluded) and rel_top (included, ACTUAL values in multiples of the Safe
    Limit), with at most ~max_ticks ticks above 1L (same logic as
    plot_radial_synthesis._regular_radial_ticks)."""
    span = rel_top - 1
    if not np.isfinite(rel_top) or span <= 0:
        return [1]
    raw = span / max_ticks
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
    k_min = int(np.floor(1 / step + 1e-9)) + 1
    k_max = int(np.floor(rel_top / step + 1e-9))
    return [1] + [round(float(k * step), 10) for k in range(k_min, k_max + 1)]


def _draw_bar_value_axis(axis, ticks, positions):
    """"nL" axis right above a bar (show_value_overshoot): ticks at the
    abscissas `positions` (bar coordinates), labeled with the actual values
    `ticks` (e.g. 1L, 100L, 1000L)."""
    x_max = axis.get_xlim()[1]
    kept = [(t, x) for t, x in zip(ticks, positions) if x is not None and 0 <= x <= x_max * (1 + 1e-9)]
    axis.spines["top"].set_visible(True)
    axis.spines["top"].set_position(("axes", 1.0))
    axis.spines["top"].set_linewidth(0.8)
    axis.spines["top"].set_color("#555555")
    axis.xaxis.tick_top()
    axis.set_xticks([x for _, x in kept])
    axis.set_xticklabels([f"{t:g}L" for t, _ in kept], fontsize=VALUE_AXIS_TICK_LABEL_FONTSIZE,
                         fontweight="bold", color="#333333")
    axis.tick_params(axis="x", top=True, labeltop=True, length=3, pad=1.5, color="#555555",
                     labelsize=VALUE_AXIS_TICK_LABEL_FONTSIZE)


def _plot_sensitivity_segment(ax, x_min, x_max, y_pos, color, cap_half_height=0.04,
                               linewidth=2.2, zorder=1.5):
    """Draw the confidence interval (sensitivity to the share variants) of a point: horizontal segment between `x_min`
    and `x_max`, with a small vertical tick at each end (error bar style).
    Drawn below the bubble (zorder < 2), which therefore hides the center of
    the segment and only lets its ends show."""
    if x_max is None or x_min is None or x_max <= x_min:
        return
    ax.plot([x_min, x_max], [y_pos, y_pos], color=color, linewidth=linewidth,
            solid_capstyle="butt", zorder=zorder)
    for x in (x_min, x_max):
        ax.plot([x, x], [y_pos - cap_half_height, y_pos + cap_half_height], color=color,
                linewidth=linewidth, zorder=zorder)


def _draw_lp_title_above(ax, name, gap_below_bar, fontsize=13):
    """Display the name of the LP above the bar, left-aligned on the start of
    the axis (x=0), anchored by its bottom (va="bottom") at `gap_below_bar`
    above the bar it names (y=1.0 = top of this bar)."""
    ax.text(0.0, 1.0 + gap_below_bar, name, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=fontsize, fontweight="bold", clip_on=False)


# ============================================================================
# FIGURE: create_multi_regions_synthesis
# ============================================================================

# Size of the scenario bubbles (area, in points^2 -- `s` parameter of
# ax.scatter), small enough for the bubbles not to overlap when a row has many
# scenarios.
BUBBLE_MARKER_SIZE_REGION = 210

# Minimum bar height per scenario (same units as row_height_factor): instead of
# a fixed bar height whatever the number of scenarios (which squeezes the y
# levels all the more as scenario_names is long), the bar height grows with
# len(scenario_names) as soon as this minimum exceeds the default height --
# which proportionally spreads the bubbles and the code labels (displayed to
# the left of each bar) of neighboring scenarios. No effect for the
# "historical" number of scenarios (2-4), where the default height remains
# larger than this minimum.
MIN_ROW_HEIGHT_PER_SCENARIO_REGION = 0.30

# Cap of row_height_factor (hence of MIN_ROW_HEIGHT_PER_SCENARIO_REGION *
# number of scenarios): the total figure height is ~row_height_factor * n_lp,
# so without a cap, an unusually large number of scenarios (e.g. a data folder
# containing one row per EXIOBASE country instead of only the expected
# FR/EU27/W) would make a figure several hundred inches tall -- enough for
# matplotlib to fail to render it correctly (blank page) instead of just
# squeezing the bubbles. Beyond this cap, the bubbles get squeezed again, but
# the figure is at least generated.
MAX_ROW_HEIGHT_FACTOR_REGION = 3.5


def _find_region_scenario_idx(scenario_names, region_codes):
    """Index of the reference scenario (region code = config.DEFAULT_REGION_CODE,
    "FR") in scenario_names, used to sort the LPs by CBA overshoot (see
    _sort_indicators_by_region_overshoot). If no scenario has this code,
    falls back to config.REFERENCE_SCENARIO_IDX."""
    for idx, name in enumerate(scenario_names):
        if str(region_codes.get(name, "")).upper() == config.DEFAULT_REGION_CODE:
            return idx
    return config.REFERENCE_SCENARIO_IDX


def _union_indicators(all_scenarios_data):
    """List of the indicators (rows of the figure) present in AT LEAST ONE
    scenario, in their order of first appearance.

    Not limited to the keys of a single reference scenario: the scenarios are
    loaded from files that may be partial for a given indicator (data
    pipeline still being written, LP missing for a geography...) -- taking a
    single scenario as the reference would make the whole figure disappear if
    that one happens to be incomplete, or would hide an LP that other scenarios
    did compute."""
    indicators = []
    seen = set()
    for scenario_data in all_scenarios_data:
        for name in scenario_data.keys():
            if name not in seen:
                seen.add(name)
                indicators.append(name)
    return indicators


def _sort_indicators_by_region_overshoot(indicators, scenario_names, all_scenarios_data,
                                            budget_of, region_codes):
    """Sort the LPs (rows of the figure) by increasing overshoot of the France
    scenario (max of CBA and PBA, if PBA is present): the further right the
    rightmost France bubble of a bar, the lower this bar is displayed relative
    to the others.

    budget_of(indicator_name, scenario_name) gives the budget ("1L") to
    compare the footprint with -- it depends on the scenario, each country
    having its own share of the world budget."""
    region_idx = _find_region_scenario_idx(scenario_names, region_codes)
    region_data = all_scenarios_data[region_idx]
    region_name = scenario_names[region_idx]

    def sort_key(indicator_name):
        payload = region_data.get(indicator_name)
        lb_val = budget_of(indicator_name, region_name)
        if payload is None or not lb_val:
            return float("-inf")
        rel_cba = _total_footprint(payload) / lb_val
        abs_pba = payload.get("pba")
        rel_pba = abs_pba / lb_val if abs_pba is not None else float("-inf")
        rel_max = max(rel_cba, rel_pba)
        return rel_max if np.isfinite(rel_max) else float("-inf")

    return sorted(indicators, key=sort_key)


def _relative_bounds(abs_value, budget_min, budget_max):
    """Bounds (left, right) of the sensitivity segment of a point: overshoot of
    `abs_value` under the two bounding budgets.

    The two results are reordered as (min, max) because a smaller budget gives
    a larger overshoot: the "min" bound of a budget is the right bound of the
    segment, not its left bound.
    """
    if abs_value is None or not np.isfinite(abs_value):
        return None, None
    variants = [abs_value / b for b in (budget_min, budget_max) if b]
    if len(variants) < 2:
        return None, None
    return min(variants), max(variants)


def _plot_region_bubble(ax, rel_x, y_pos, style, hatched=False, hatch_color=None):
    """Draw a scenario bubble on `ax` (size BUBBLE_MARKER_SIZE_REGION).

    hatched=True: hatched bubble (PBA accounting); hatch_color controls its
    color (white for the "trend" scenarios, to stay readable on a dark background)."""
    sc = ax.scatter(rel_x, y_pos, s=BUBBLE_MARKER_SIZE_REGION, marker="o", facecolor=style["face"],
                     edgecolor=style["edge"], linewidth=1.5, zorder=2)
    if hatched:
        sc.set_hatch("///")
        if hatch_color is not None:
            sc.set_edgecolor(hatch_color)


def create_multi_regions_synthesis(
    all_scenarios_data, boundaries_df, shares_df, scenario_names,
    unit_row=config.UNIT_ROW_ABS,
    ub_color="#bc270a", x_main_max=8.0, display_bounds=True,
    show_pba=False, show_value_overshoot=False,    title_gap_below_bar=0.01, title_gap_above_bar=0.01,
    title="Overshoot relative to the Safe Operating Space",
    sharing_principle="EPC", sensitivity=False,
    region_codes=None, default_region_code=config.DEFAULT_REGION_CODE,
    pop_df=None, scale_UB=False,
    excel_path=None,
):
    """Multi-LP Overshoot / Safe Operating Space figure, in absolute values and
    with a budget specific to each country/region. X axis relative to the
    budget of each scenario (1L = Safe Boundary).

    Nothing is divided by the population: neither the footprints (to be given
    in absolute values, see processing.process_scenario_absolute), nor the
    budgets. The world budgets remain those of boundaries_values.xlsx/sheet "All"
    ("Lower Safe Boundary"/"Safe Boundary"/"High-Risk Boundary", converted by "Unit
    conversion budget"), and the descent to the national level is done by
    multiplying them by the share of the country read in
    budget_shares.xlsx/sheet "Budget_shares_regions":

        country_budget = World_budget / Budget_conversion * Country_share

    The country/region of each scenario is deduced from the name of its data
    folder through the "EXIOBASE code" column ("2019_EU27" -> row "EU27",
    "2019_FR" -> "FR", "2019_W" -> "W"); see processing.resolve_region_code.

    The bubble color and the identification of the scenarios go through the
    resolved region code (colors.region_code_style_map):
    "FR" keeps the light blue, "EU27" the red, "W" the green
    (colors.REGION_CODE_COLORS), any other code a generic grey -- two scenarios
    with different names attached to the same code (e.g. Base_year and TREND,
    both "FR" by default) therefore get the same color. The legend has no entry
    per scenario: each scenario is identified by its code, displayed as text to
    the left of each bar, vertically aligned on its bubble.

    Consequence: the 1L abscissa is no longer the same budget in absolute value
    for all the scenarios (each bubble is positioned relative to the budget of
    its own country), whereas the Lower/High-Risk Boundary lines remain common to
    the whole row -- their relative position (UB/LB) does not depend on the
    share, which cancels out in the ratio.

    Args:
        shares_df: "Budget_shares_regions" sheet of budget_shares.xlsx
            (io.load_budget_shares_df).
        unit_row: display unit row of boundaries_df -- config.UNIT_ROW_ABS by
            default, the figure being in absolute values.
        sharing_principle: "EPC" (equal per capita) or "CTR" (capability to
            reduce). The reference share, which positions the bubbles, is read
            in the "CTR share ref. ..." column for CTR and, for EPC, in "UN –
            EPC 2019" ("UN – EPC 2019-2100" for GHG emissions) -- see
            config.region_figure_epc_ref_pattern. The sensitivity bounds remain
            "<principle> min/max". The world budgets of GHG emissions are read
            in the "GHG emissions (2019)" column of boundaries_values.xlsx
            (config.region_figure_boundary_column).
        sensitivity: if True, each bubble gets a horizontal segment (confidence
            interval, with small vertical ticks at the ends) connecting the two
            overshoot levels obtained with the budgets of the "<principle> min"
            and "<principle> max" columns. The segment is white for the CBA
            bubbles (solid) and black for the PBA bubbles (hatched,
            show_pba=True). A min share giving a smaller budget, hence a larger
            overshoot, it sets the RIGHT end of the segment (and the max share
            the left end).
            W (share 1, no min/max): if pop_df is given, the segment is
            obtained per capita -- world limit divided by the World population
            of each period of config.WORLD_SENSITIVITY_POPULATION_PERIODS (2019,
            mean 2019-2050, mean 2019-2100), compared with the world per-capita
            pressures (2019 population). The reference bubble is unchanged (see
            processing.compute_world_period_budget).
        pop_df: "Population" sheet of budget_shares.xlsx (io.load_population_df),
            only required for the sensitivity of W.
        scale_UB: if False (default), common scale for all the rows, in
            multiples of the lower limit (1L, 2L, 3L...). If True, the scale of
            each row is specific to its indicator, piecewise linear:
            unchanged between 0 and the Safe Boundary (1L), then stretched or
            compressed so that the High-Risk Boundary falls exactly on the
            "Increasing Risk" / "High-Risk Zone" boundary (formerly 2L); beyond
            the UB, the slope of the [Safe Boundary, UB] segment is extended. The
            Safe Boundary and High-Risk Boundary lines are thus vertically aligned
            from one row to another, and all the abscissas of the row (bubbles,
            sensitivity segments, Lower Safe Boundary) follow this scale. The "nL"
            ticks (main axis and break axis), which no longer have a common
            meaning, are hidden. No axis break: the main axis takes the full
            width, up to x_main_max, widened if needed (same bound for all the
            rows) to contain the rightmost bubble of the figure. A row without
            High-Risk Boundary (or with UB <= Safe Boundary) keeps the 1L scale.
        show_value_overshoot: if True, each bar gets its own "nL" axis right
            above it, instead of the single axis at the top of the figure (the
            title of the bar is shifted above this axis). With scale_UB=True,
            this axis follows the scale specific to the row: 1L, then ticks
            with a "round" step (e.g. 1L, 100L, 200L... or 1L, 2.5L, 5L...) in
            actual values, positioned according to the UB-rescaled scale --
            like the per-sector ticks of the radial figure. Without scale_UB:
            integer ticks 1L, 2L, 3L...
        region_codes: optional, {scenario_name: EXIOBASE code} to force the
            attachment of a scenario to a row of shares_df, instead of
            deducing it from its name.
        default_region_code: code used for a scenario whose name contains no
            EXIOBASE code ("FR" by default: Base_year, TREND, Tech_NZE... are
            France scenarios).
        display_bounds: True (default) displays the 3 lines (Lower Safe Boundary,
            Safe Boundary, High-Risk Boundary); False only displays Safe Boundary.
        show_pba: if True, also draws a hatched bubble from the "pba" key of
            each payload (production-based accounting).
        The LP name is displayed above each bar, left-aligned (no unit, no
            Safe Boundary value).
        title_gap_below_bar, title_gap_above_bar: gaps, as a fraction of bar height, between the title and the bar it
            names / the bar above.
        excel_path: if given, Excel workbook of the plotted data (see
            figure_export), "Overshoot" sheet: one row per indicator x
            scenario x accounting with the footprint, the budgets of the
            region (Lower Safe Boundary, Safe Boundary, High-Risk Boundary), the
            overshoot of the Safe Boundary (actual value, not rescaled by
            scale_UB) and the sensitivity bounds.

    Returns:
        matplotlib.figure.Figure
    """
    if not all_scenarios_data:
        raise ValueError("all_scenarios_data is empty")
    if sharing_principle not in config.BUDGET_SHARE_COLUMN_PATTERNS:
        raise ValueError(
            f"Unknown sharing_principle: {sharing_principle!r} "
            f"(expected: {', '.join(config.BUDGET_SHARE_COLUMN_PATTERNS)})"
        )

    codes = {
        name: (region_codes or {}).get(
            name, processing.resolve_region_code(name, shares_df, default=default_region_code)
        )
        for name in scenario_names
    }

    # Display order of the scenarios (top to bottom): FR, EU27, W, then the other
    # regions in alphabetical order of their displayed name. Stable sort: the
    # scenarios sharing a region (Base_year, TREND... -> FR) keep their order.
    def _region_label(code):
        if str(code).upper() == config.WORLD_REGION_CODE:
            return "World"
        if str(code).upper() == "EU27":
            return "EU27"
        if (config.BUDGET_SHARE_REGION_COLUMN in shares_df.columns
                and code in shares_df.index):
            return str(shares_df.loc[code, config.BUDGET_SHARE_REGION_COLUMN])
        return str(code)

    _fixed_order = {config.DEFAULT_REGION_CODE: 0, "EU27": 1, config.WORLD_REGION_CODE: 2}
    # all_scenarios_data is aligned by position on scenario_names: sort both together.
    order = sorted(
        range(len(scenario_names)),
        key=lambda i: (_fixed_order.get(str(codes[scenario_names[i]]).upper(), 3),
                       _region_label(codes[scenario_names[i]]).lower()
                       if str(codes[scenario_names[i]]).upper() not in _fixed_order else ""),
    )
    scenario_names = [scenario_names[i] for i in order]
    all_scenarios_data = [all_scenarios_data[i] for i in order]

    # Reference EPC read in the UN columns ("UN – EPC 2019", or "UN – EPC
    # 2019-2100" for GHG): see config.region_figure_epc_ref_pattern.
    if sharing_principle == "EPC":
        for pattern in {config.REGION_FIGURE_EPC_REF_PATTERN,
                        *config.REGION_FIGURE_EPC_REF_PATTERN_BY_INDICATOR.values()}:
            if not any(re.search(pattern, str(c), flags=re.IGNORECASE) for c in shares_df.columns):
                raise ValueError(
                    f"No column of shares_df matches {pattern!r}: regenerate "
                    f"budget_shares.xlsx (\"UN – EPC\" columns) and reload it with "
                    f"io.load_budget_shares_df."
                )

    def budget_of(indicator_name, scenario_name, boundary_kind="lb", variant="ref"):
        column_pattern = (config.region_figure_epc_ref_pattern(indicator_name)
                          if sharing_principle == "EPC" and variant == "ref" else None)
        # Column of boundaries_values.xlsx holding the budgets (e.g. GHG emissions ->
        # GHG emissions (2019)), or the indicator column if the sheet lacks it
        # ("Agri"): see config.region_figure_boundary_column.
        boundary_column = config.region_figure_boundary_column(indicator_name, boundaries_df)
        budget = processing.compute_region_budget(
            boundaries_df, shares_df, boundary_column, codes[scenario_name],
            sharing_principle, variant=variant, boundary_kind=boundary_kind,
            column_pattern=column_pattern,
        )
        # W has no min/max shares: per-capita sensitivity, by varying the
        # population period of the limit (see pop_df).
        if (budget is None and variant != "ref" and pop_df is not None
                and str(codes[scenario_name]).upper() == config.WORLD_REGION_CODE):
            period_budgets = [
                processing.compute_world_period_budget(boundaries_df, pop_df, boundary_column, period,
                                                       boundary_kind=boundary_kind)
                for period in config.WORLD_SENSITIVITY_POPULATION_PERIODS
            ]
            period_budgets = [b for b in period_budgets if b]
            if period_budgets:
                return min(period_budgets) if variant == "min" else max(period_budgets)
        return budget

    indicators = _union_indicators(all_scenarios_data)
    if not indicators:
        raise ValueError(
            "No indicator found in all_scenarios_data (all the scenarios are empty) -- "
            "check that the processing of the scenario folders did produce data."
        )
    indicators = _sort_indicators_by_region_overshoot(
        indicators, scenario_names, all_scenarios_data, budget_of, codes,
    )
    n_lp = len(indicators)

    scenario_style = colors.region_code_style_map(codes)
    y_levels = np.linspace(0.82, 0.18, len(scenario_names)) if len(scenario_names) > 1 else np.array([0.5])
    scenario_to_y = dict(zip(scenario_names, y_levels))

    # Bar height: the default value (1.35/1.2), unless the number of scenarios
    # is large enough to require more (MIN_ROW_HEIGHT_PER_SCENARIO_REGION),
    # capped at MAX_ROW_HEIGHT_FACTOR_REGION -- see their comments. No effect on
    # the figures with 2-4 scenarios.
    default_row_height_factor = 1.35
    row_height_factor = max(
        default_row_height_factor,
        min(MIN_ROW_HEIGHT_PER_SCENARIO_REGION * len(scenario_names), MAX_ROW_HEIGHT_FACTOR_REGION),
    )
    # hspace sized so that the title block (name + unit) always fits between
    # the two requested gaps, whatever their settings (+ the "nL" axis
    # specific to each bar if show_value_overshoot).
    value_axis_extra = VALUE_AXIS_BLOCK_HEIGHT if show_value_overshoot else 0.0
    hspace = title_gap_below_bar + title_gap_above_bar + TITLE_BLOCK_HEIGHT + value_axis_extra
    fig = plt.figure(figsize=(18, max(8, row_height_factor * n_lp + 1.8)))
    gs = fig.add_gridspec(n_lp, 2, width_ratios=[7.0, 3.0], hspace=hspace, wspace=0.02)

    def compute_row(indicator_name):
        """Budgets, limits and positions (rescaled if scale_UB) of the bubbles
        of a row, or None if no budget is available."""
        # One budget per scenario (= per country): the reference budget is
        # used to position the bubbles of this scenario, and the min/max
        # budgets to bound the sensitivity segment.
        budgets = {name: budget_of(indicator_name, name) for name in scenario_names}
        if all(b is None for b in budgets.values()):
            return None

        # The Lower/High-Risk Boundary lines are common to the row: their
        # relative position (World_budget_LOWER/UB / World_budget_LB) is the
        # same for all the countries, the share cancelling out in the ratio.
        # They are therefore computed with the budget of the reference scenario.
        ref_name = next(name for name in scenario_names if budgets[name])
        ref_lb = budgets[ref_name]
        lower_val = budget_of(indicator_name, ref_name, boundary_kind="lower")
        ub_val = budget_of(indicator_name, ref_name, boundary_kind="ub")
        lower_rel = (lower_val / ref_lb) if lower_val is not None else None
        ub_rel = (ub_val / ref_lb) if ub_val is not None else None

        # scale_UB=True: scale specific to the row, piecewise linear, which
        # keeps the Safe Boundary (1L) on config.SAFE_BOUNDARY_REL and brings the
        # High-Risk Boundary to config.RISK_TRANSITION_REL -- the two lines are
        # therefore vertically aligned from one row to another. Below 1L, the
        # scale is unchanged; beyond, the slope of the [1L, UB] segment is
        # extended. Applied to all the abscissas of the row.
        ub_slope = None
        if scale_UB and ub_rel is not None and np.isfinite(ub_rel) and ub_rel > 1:
            ub_slope = (config.RISK_TRANSITION_REL - config.SAFE_BOUNDARY_REL) / (ub_rel - 1)

        def scaled(x):
            if x is None or ub_slope is None:
                return x
            if x <= 1:
                return x * config.SAFE_BOUNDARY_REL
            return config.SAFE_BOUNDARY_REL + (x - 1) * ub_slope

        def unscaled(x):
            """Inverse of scaled (displayed abscissa -> actual value in nL)."""
            if x is None or ub_slope is None:
                return x
            if x <= config.SAFE_BOUNDARY_REL:
                return x / config.SAFE_BOUNDARY_REL
            return 1 + (x - config.SAFE_BOUNDARY_REL) / ub_slope

        # Relative positions (CBA, and PBA if requested) by scenario
        points = []
        for scenario_idx, scenario_data in enumerate(all_scenarios_data):
            sname = scenario_names[scenario_idx]
            lb_val = budgets[sname]
            if indicator_name not in scenario_data or not lb_val:
                continue
            payload = scenario_data[indicator_name]
            abs_cba = _total_footprint(payload)
            rel_cba = abs_cba / lb_val
            if not np.isfinite(rel_cba):
                continue

            abs_pba = payload.get("pba") if show_pba else None
            rel_pba = abs_pba / lb_val if (abs_pba is not None and np.isfinite(abs_pba)) else None

            rel_cba_min = rel_cba_max = None
            rel_pba_min = rel_pba_max = None
            budget_min = budget_max = None
            if sensitivity:
                budget_min = budget_of(indicator_name, sname, variant="min")
                budget_max = budget_of(indicator_name, sname, variant="max")
                rel_cba_min, rel_cba_max = _relative_bounds(abs_cba, budget_min, budget_max)
                if abs_pba is not None:
                    rel_pba_min, rel_pba_max = _relative_bounds(abs_pba, budget_min, budget_max)

            points.append({
                "sname": sname, "y_pos": scenario_to_y[sname],
                "abs_cba": abs_cba, "rel_cba": scaled(rel_cba),
                "abs_pba": abs_pba, "rel_pba": scaled(rel_pba),
                "rel_cba_min": scaled(rel_cba_min), "rel_cba_max": scaled(rel_cba_max),
                "rel_pba_min": scaled(rel_pba_min), "rel_pba_max": scaled(rel_pba_max),
                # Actual overshoot levels (not rescaled by scale_UB) and
                # sensitivity budgets, for the Excel export (excel_path)
                "raw": {"CBA": (abs_cba, rel_cba, rel_cba_min, rel_cba_max),
                        "PBA": (abs_pba, rel_pba, rel_pba_min, rel_pba_max)},
                "budget_min": budget_min, "budget_max": budget_max,
            })

        return {"budgets": budgets, "ref_lb": ref_lb, "lower_rel": lower_rel, "ub_rel": ub_rel,
                "scaled": scaled, "unscaled": unscaled, "points": points}

    rows = {name: compute_row(name) for name in indicators}

    # scale_UB=True: no axis break. The main axis takes the full width, and its
    # right bound is common to all the rows (to keep the Safe Boundary/UB lines
    # aligned): x_main_max, widened if needed to contain the rightmost bubble
    # (or end of sensitivity segment) of the figure.
    x_axis_max = x_main_max
    if scale_UB:
        all_row_x = [
            p[key] for row in rows.values() if row is not None for p in row["points"]
            for key in ("rel_cba", "rel_pba", "rel_cba_max", "rel_pba_max") if p[key] is not None
        ]
        if all_row_x:
            x_axis_max = max(x_main_max, max(all_row_x) * 1.04)

    top_main_ax = None
    value_axes = []

    for row_idx, indicator_name in enumerate(indicators):
        ax = fig.add_subplot(gs[row_idx, :] if scale_UB else gs[row_idx, 0])
        ax_ext = fig.add_subplot(gs[row_idx, 1], sharey=ax)
        ax.set_zorder(3)
        ax_ext.set_zorder(2)
        ax_ext.patch.set_alpha(0.0)
        if row_idx == 0:
            top_main_ax = ax

        row = rows[indicator_name]
        if row is None:
            ax.text(0.5, 0.5, f"{indicator_name}: missing lower limit", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="#7a7a7a")
            ax.set_xlim(0, x_axis_max)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax_ext.set_visible(False)
            continue
        budgets, ref_lb, points, scaled = row["budgets"], row["ref_lb"], row["points"], row["scaled"]
        lower_rel, ub_rel = row["lower_rel"], row["ub_rel"]

        # Axis overflow: gathers ALL the x values present on this row (not just
        # hard-coded cba/pba), to stay correct if another series is added one
        # day. Never a break with scale_UB=True.
        all_x = [p["rel_cba"] for p in points] + [p["rel_pba"] for p in points if p["rel_pba"] is not None]
        overflow = [x for x in all_x if x > x_axis_max]
        use_break = len(overflow) > 0 and not scale_UB

        ax.set_xlim(0, x_axis_max)
        ax.set_ylim(0, 1)
        # Zones bounded by the limits of the row: Safe Operating Space up to the
        # Safe Boundary, Increasing Risk up to the High-Risk Boundary, High-Risk beyond.
        _ub_x = scaled(ub_rel) if (ub_rel is not None and np.isfinite(ub_rel) and ub_rel > 1) else None
        if use_break:
            x2_min = min(overflow) * 0.995
            x2_max = max(overflow) * 1.02
            if (x2_max - x2_min) < 0.5:
                mid = 0.5 * (x2_min + x2_max)
                x2_min, x2_max = mid - 0.25, mid + 0.25
        _draw_lp_background(ax, x_axis_max, safe_x=scaled(1.0), risk_x=_ub_x)

        if use_break:
            ax_ext.set_xlim(x2_min, x2_max)
            ax_ext.set_ylim(0, 1)
            _draw_break_background(ax_ext, x2_min, x2_max, x_axis_max,
                                   safe_x=scaled(1.0), risk_x=_ub_x)
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

        lower_rel = scaled(lower_rel)
        ub_rel = scaled(ub_rel)
        safe_x = scaled(1.0)

        # Limit lines (independent of the background gradient): Safe Limit
        # (green, historical boundary) always displayed; Lower/Upper Safe Bound
        # (light green/red) only if display_bounds=True. No values displayed.
        if display_bounds and lower_rel is not None and lower_rel >= 0:
            ax.axvline(lower_rel, ymin=0.10, ymax=0.90, color=config.LOWER_SAFE_BOUNDARY_COLOR, linewidth=3, zorder=3)
        ax.axvline(safe_x, ymin=0.10, ymax=0.90, color="#0d9a33", linewidth=3, zorder=3)
        if display_bounds and ub_rel is not None and ub_rel <= x_axis_max:
            ax.axvline(ub_rel, ymin=0.10, ymax=0.90, color=ub_color, linewidth=3, zorder=3)
        elif (display_bounds and use_break and ub_rel is not None
              and ax_ext.get_xlim()[0] <= ub_rel <= ax_ext.get_xlim()[1]):
            ax_ext.axvline(ub_rel, ymin=0.10, ymax=0.90, color=ub_color, linewidth=3, zorder=3)

        # CBA bubbles (solid) + PBA (hatched, if show_pba), with sensitivity
        # segment (sensitivity=True) drawn below each bubble.
        for p in points:
            style = scenario_style.get(p["sname"], colors.REGION_CODE_FALLBACK)

            in_ext_cba = use_break and p["rel_cba"] > x_axis_max and ax_ext.get_visible()
            target_ax_cba = ax_ext if in_ext_cba else ax
            if p["rel_cba_min"] is not None:
                _plot_sensitivity_segment(target_ax_cba, p["rel_cba_min"], p["rel_cba_max"], p["y_pos"],
                                           color="white")
            _plot_region_bubble(target_ax_cba, p["rel_cba"], p["y_pos"], style, hatched=False)

            if p["rel_pba"] is not None:
                hatch_color = "white" if "trend" in p["sname"].lower() else "black"
                in_ext_pba = use_break and p["rel_pba"] > x_axis_max and ax_ext.get_visible()
                target_ax_pba = ax_ext if in_ext_pba else ax
                if p["rel_pba_min"] is not None:
                    _plot_sensitivity_segment(target_ax_pba, p["rel_pba_min"], p["rel_pba_max"], p["y_pos"],
                                               color="black")
                _plot_region_bubble(target_ax_pba, p["rel_pba"], p["y_pos"], style, hatched=True,
                                     hatch_color=hatch_color)

        # Code labels to the left of the bar: replace the "1 entry per
        # scenario" legend by a short label (the region code) aligned on the
        # bubble of each scenario, repeated on each row since the y position of
        # a scenario is common to the whole figure. x as a fraction of the
        # axes, y in data coordinates (get_yaxis_transform), to stay just to
        # the left of x=0 whatever x_main_max.
        yaxis_transform = ax.get_yaxis_transform()
        for sname, y_pos in scenario_to_y.items():
            style = scenario_style.get(sname, colors.REGION_CODE_FALLBACK)
            # Region name from the "Region" column of budget_shares.xlsx
            # ("France", "RoW Africa"...); W is named "World".
            code_text = _region_label(codes.get(sname, sname))
            ax.text(-0.012, y_pos, code_text, transform=yaxis_transform, ha="right", va="center",
                    fontsize=11, fontweight="bold", color=style["face"], clip_on=False)

        # LP name + unit above the bar. `ceiling` (= 1.0 + hspace, the bottom of the
        # bar above) is the same formula for all the rows, including the 1st:
        # hspace is sized (see above) to also leave room for the ticks of the
        # shared X axis above it.
        _draw_lp_title_above(ax, indicator_name, title_gap_below_bar + value_axis_extra)

        # Clean styling
        for axis in (ax, ax_ext):
            if not axis.get_visible():
                continue
            axis.set_yticks([])
            axis.grid(False)
            for spine in ("left", "right", "bottom"):
                axis.spines[spine].set_visible(False)
            axis.spines["top"].set_visible(
                not scale_UB and ((axis is ax and row_idx == 0) or (axis is ax_ext and use_break))
            )
            axis.xaxis.tick_top()
            axis.tick_params(axis="x", length=4, pad=4)
            if scale_UB:
                # Scale specific to each row: no common "nL" ticks.
                axis.set_xticks([])

        if show_value_overshoot:
            # "nL" axis specific to each bar, right above it. With scale_UB:
            # ticks with a "round" step in actual values (from 1L to the actual
            # value at the right edge of the axis), positioned by scaled() --
            # like the per-sector ticks of the radial figure.
            if scale_UB:
                ticks = _regular_rel_ticks(row["unscaled"](x_axis_max))
                positions = [scaled(t) for t in ticks]
            else:
                ticks = list(range(1, int(x_axis_max) + 1))
                positions = ticks
            _draw_bar_value_axis(ax, ticks, positions)
            value_axes.append(ax)
            if use_break and ax_ext.get_visible():
                ax_ext.spines["top"].set_position(("axes", 1.0))
                ax_ext.tick_params(axis="x", labeltop=True, top=True, labelsize=VALUE_AXIS_TICK_LABEL_FONTSIZE,
                                   length=3, pad=1.5)
                value_axes.append(ax_ext)
        else:
            if row_idx != 0:
                ax.tick_params(axis="x", labeltop=False, top=False)
            if use_break and ax_ext.get_visible():
                ax_ext.tick_params(axis="x", labeltop=True, top=True, labelsize=AXIS_TICK_LABEL_FONTSIZE, length=3, pad=2)

    # Common main X axis (1L, 2L, 3L...) + zone labels, on the 1st row
    # (scale_UB=True or show_value_overshoot=True: zone labels only -- no
    # common "nL" axis)
    if top_main_ax is not None:
        top_spine_y = 1.0 + hspace + 0.03
        if not scale_UB and not show_value_overshoot:
            top_main_ax.set_xticks(np.arange(1, int(x_main_max) + 1))
            top_main_ax.set_xticklabels([f"{i}L" for i in range(1, int(x_main_max) + 1)], fontsize=AXIS_TICK_LABEL_FONTSIZE, fontweight="bold")
            top_main_ax.spines["top"].set_position(("axes", top_spine_y))
            top_main_ax.tick_params(axis="x", pad=8, length=5, labelsize=AXIS_TICK_LABEL_FONTSIZE)
        if show_value_overshoot:
            zone_label_y = 1.0 + hspace + 0.05
        else:
            zone_label_y = top_spine_y + 0.45
        # Each label starts at the beginning of its zone (x in data
        # coordinates): 0, Safe Boundary (SAFE_BOUNDARY_REL), High-Risk Boundary
        # (RISK_TRANSITION_REL -- where the UB falls on every row with scale_UB).
        # Constraints: "Safe Operating Space" ends just before 1L, "Increasing
        # Risk" starts just after 1L, "High-Risk Zone" starts just after the
        # UB (RISK_TRANSITION_REL) -- and after the end of "Increasing Risk"
        # if needed, so that no label overlaps.
        # Positions depend on the size of the axes in pixels: recomputed after
        # tight_layout (see _place_zone_labels below).
        text_kw = dict(transform=top_main_ax.get_xaxis_transform(), fontsize=11,
                       fontweight="bold", clip_on=False)
        zone_labels = (
            top_main_ax.text(0, zone_label_y, "Safe Operating Space", color="#0b7d3e", ha="right", **text_kw),
            top_main_ax.text(0, zone_label_y, "Increasing Risk", color="#d47818", ha="left", **text_kw),
            top_main_ax.text(0, zone_label_y, "High-Risk Zone", color="#7a1f54", ha="left", **text_kw),
        )

        def _place_zone_labels():
            safe_txt, increasing_txt, high_txt = zone_labels
            margin_px = 6 * fig.dpi / 72  # small gap (6 pt) to a limit / another label
            to_px = top_main_ax.transData.transform
            to_data = top_main_ax.transData.inverted().transform
            safe_px = to_px((config.SAFE_BOUNDARY_REL, 0))[0]
            safe_txt.set_x(to_data((safe_px - margin_px, 0))[0])
            increasing_txt.set_x(to_data((safe_px + margin_px, 0))[0])
            increasing_end_px = increasing_txt.get_window_extent(renderer=fig.canvas.get_renderer()).x1
            high_px = max(to_px((config.RISK_TRANSITION_REL, 0))[0], increasing_end_px) + margin_px
            high_txt.set_x(to_data((high_px, 0))[0])

        _place_zone_labels()

    # Legend: limits + type of accounting (show_pba) + sensitivity
    # (sensitivity) -- no entry per scenario, each scenario is identified by
    # its code to the left of each bar (see above).
    limit_handles = [Line2D([0], [0], color="#0d9a33", linewidth=3.5, label="Safe Boundary")]
    if display_bounds:
        limit_handles = [
            Line2D([0], [0], color=config.LOWER_SAFE_BOUNDARY_COLOR, linewidth=3.5, label="Lower Safe Boundary"),
            *limit_handles,
            Line2D([0], [0], color=ub_color, linewidth=3.5, label="High-Risk Boundary"),
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

    # Title and legend flush against the figure edges; the layout rectangle is
    # then deduced from their measured height, to leave no blank space.
    legend = fig.legend(handles=legend_handles, loc="lower center", ncol=min(8, len(legend_handles)),
                        bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=13,
                        borderaxespad=0.1)
    suptitle = fig.suptitle(title, fontsize=17, fontweight="bold", y=1.0, va="top")
    left_rect = 0.03
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_h_px = fig.bbox.height
    bottom_rect = legend.get_window_extent(renderer).height / fig_h_px
    top_rect = 1.0 - suptitle.get_window_extent(renderer).height / fig_h_px
    # pad=0.1 (default 1.08): tight_layout's outer margin, which left a blank
    # between the title/legend and the plot; h_pad/w_pad keep the inner spacing.
    plt.tight_layout(pad=0.1, h_pad=1.08, w_pad=1.08,
                     rect=[left_rect, bottom_rect, 0.98, top_rect])
    if top_main_ax is not None:
        # tight_layout() regenerates the labels of the main axis (spine moved
        # higher) and makes them lose the bold applied via set_xticklabels; it
        # is therefore reapplied at the very end.
        plt.setp(top_main_ax.get_xticklabels(), fontweight="bold")
        _place_zone_labels()
    for axis in value_axes:
        plt.setp(axis.get_xticklabels(), fontweight="bold")

    if excel_path is not None:
        overshoot_rows = []
        # Same order as the rows of the figure (top to bottom)
        for indicator_name in indicators:
            row = rows[indicator_name]
            if row is None:
                continue
            unit = figure_export.indicator_unit(boundaries_df, indicator_name, unit_row)
            for p in row["points"]:
                sname = p["sname"]
                budgets_row = {
                    "Lower Safe Boundary": budget_of(indicator_name, sname, boundary_kind="lower"),
                    "Safe Boundary": row["budgets"][sname],
                    "High-Risk Boundary": budget_of(indicator_name, sname, boundary_kind="ub"),
                }
                for accounting, (value, rel, rel_min, rel_max) in p["raw"].items():
                    if value is None:
                        continue
                    overshoot_rows.append({
                        "Indicator": indicator_name, "Unit": unit, "Scenario": sname,
                        "Region code": codes[sname], "Accounting": accounting,
                        "Footprint": figure_export.clean_value(value),
                        **{label: figure_export.clean_value(b) for label, b in budgets_row.items()},
                        "Overshoot of the Safe Boundary (L)": figure_export.clean_value(rel),
                        "Sensitivity: min share budget": figure_export.clean_value(p["budget_min"]),
                        "Sensitivity: max share budget": figure_export.clean_value(p["budget_max"]),
                        "Sensitivity: overshoot min (L)": figure_export.clean_value(rel_min),
                        "Sensitivity: overshoot max (L)": figure_export.clean_value(rel_max),
                    })
        epc_note = ("UN – EPC 2019 (UN – EPC 2019-2100 and GHG emissions (2019) budgets for GHG emissions)"
                    if sharing_principle == "EPC" else "reference share")
        figure_export.write_figure_excel(excel_path, {"Overshoot": overshoot_rows}, about={
            "Figure": title,
            "Generated by": "planefr_lib.plot_overshoot.create_multi_regions_synthesis",
            "Content": "Footprint of each country/region (absolute values) compared with its own budget: "
                       "world budget x share of the region; overshoot = footprint / Safe Boundary (1L).",
            "Accounting": "CBA = consumption-based" + (", PBA = production-based" if show_pba else ""),
            "Sharing principle": f"{sharing_principle} ({epc_note}, budget_shares.xlsx)",
            "Sensitivity": ("Budgets with the min/max shares of the principle; World: per-capita limit over "
                            "the population periods " + ", ".join(config.WORLD_SENSITIVITY_POPULATION_PERIODS)
                            if sensitivity else None),
            "Unit": "Unit column (\"Figures unit\" of boundaries_values.xlsx)",
        })

    return fig
