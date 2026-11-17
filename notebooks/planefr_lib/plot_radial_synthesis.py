"""
"Radial wheel of planetary boundaries" figure: 1 sector (wedge) per LP, with
1 radial bar per scenario (height = value / Safe Limit, so the Safe Limit is a
circle of constant radius = 1L), an annotated outer ring (1 sub-band per
scenario) and an LP name + unit per sector.

A single function, driven by 2 independent booleans show_cba/show_pba (both
True by default -- the figure then combines both):
  - show_cba=True: consumption values. The domestic/imported data by category
    exist for the CBA -> each bar can be stacked like create_stacked_bar_chart
    (solid dom + hatched imp, by consumption category), just folded into a
    radial instead of a Cartesian layout. show_categories and
    show_import_split (bool, independent) enable/disable each of these two
    distinctions; if both are False, the bar becomes solid again, colored with
    the color of the LP (shaded by scenario if show_pba=False, see below).
  - show_pba=True: production values (payload["pba"]). This is already a
    single scalar in the source data (process_single_indicator_scenario does
    not break it down by category or by dom/imp) -> each bar remains solid,
    colored with a lighter shade of the LP color (to distinguish it from the
    CBA when both are displayed), whatever show_categories/show_import_split
    (ignored on the PBA side).
  - show_cba=True AND show_pba=True: both accountings on the same figure --
    for each scenario, 2 adjacent bars (CBA then PBA) and 2 adjacent sub-bands
    in the outer ring.

Title and legend adapt automatically to show_cba/show_pba (no need to repeat
them at each call).

Colors of the bars/outer ring: when CBA and PBA are displayed together
(show_cba AND show_pba), each LP keeps its two shades -- solid for the CBA,
light for the PBA -- to distinguish the two accountings. When only one of the
two is displayed ("solo" figure), the two shades no longer distinguish CBA/PBA
(there is only one): they then distinguish the reference scenario (base_year,
solid shade -- the former "CBA" color) from the other scenarios (light shade --
the former "PBA" color), which all share the same color. The scenarios are
otherwise only distinguished by their angular position within the sector and
by the text of the outer ring -- for the two "opposite" LPs on the wheel (Raw
material consumption / Blue water consumption), the scenario name is also
recalled above each cell of the outer ring.
"""

import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, config, io, processing
from .plot_stacked_bar import _build_category_legend
from .processing import DOMESTIC, IMPORTED

# ============================================================================
# COLORS
# ============================================================================

LP_COLORS = {
    "N surplus":                 "#DC8968",
    "P surplus":                 "#6D4C41",
    "Blue water consumption":    "#096DB4",
    "Agricultural land":         "#3B5A17",
    "Cropland use":              "#68BB09",
    "GHG emissions":             "#F57C00",
    "Biodiversity loss - PDF":   "#B12BB1",
    "Biodiversity loss - HANPP": "#FC85FC",
    "Raw material consumption":  "#757575",
}
_FALLBACK_LP_COLOR = "#4c78a8"

LB_COLOR = "#0d9a33"    # same as plot_stacked_bar.py / plot_overshoot.py ("Safe Limit")
UB_COLOR = "#bc270a"    # same as plot_stacked_bar.py / plot_overshoot.py ("Upper Safe Bound")
LOWER_COLOR = config.LOWER_SAFE_BOUND_COLOR  # same as plot_stacked_bar.py / plot_overshoot.py ("Lower Safe Bound")

# BdF with comparison (group of 2 scenarios): outline of the gap to the base year
COMPARISON_EDGE_COLOR = "black"
COMPARISON_EDGE_WIDTH = 0.3

# A large overshoot (e.g. GHG emissions at ~16L) would squash the radial scale
# of all the sectors: with scale_diff=True, the radial ceiling is capped at this
# multiple of the Safe Limit, whatever radial_power (unless r_max is explicit).
# The bars exceeding it are truncated at the outer circle and their actual
# level ("16L") is written at their top.
SCALE_DIFF_CAP_REL = 6.0

# Margin (fraction of the maximum relative value -- bar or UB/LB/DLS threshold)
# added above this value to set the radial ceiling: the outer ring starts just
# above the highest element, without rounding up to the next whole L (which
# could leave up to 1L of dead space).
RADIAL_AXIS_MARGIN_FRAC = 0.02

# No subtitle summarizing the order of the scenarios: for these two LPs, chosen
# opposite each other on the wheel, the scenario name is recalled directly
# above the (total/variation) cell matching it in the outer ring.
SCENARIO_LABEL_INDICATORS = ("Raw material consumption", "Blue water consumption")

# scale_UB=True: common position (in multiples of the Safe Limit, before
# radial_power) to which the Upper Safe Bound of each sector is brought -- same
# ratio as the "Increasing Risk" / "High-Risk Zone" boundary of plot_overshoot
# (config.RISK_TRANSITION_REL / config.SAFE_LIMIT_REL).
UB_SCALED_REL = config.RISK_TRANSITION_REL / config.SAFE_LIMIT_REL


# ============================================================================
# POLAR GEOMETRY
# ============================================================================


def _wedge_geometry(n_lp, n_scenarios, gap_frac=0.10, bar_frac=0.82):
    """Split the circle into n_lp equal sectors (starting at 12 o'clock,
    clockwise), each leaving a visual gap (gap_frac) before spreading
    n_scenarios bars of equal width (bar_frac of the room available per bar).

    Also computes a separate geometry for the outer ring (ring_*): unlike the
    bars, the ring must leave no gap -- its segments tile the whole sector
    (full wedge_span, not usable_span) and touch from one sector to the next.

    Returns:
        dict with wedge_span/usable_span (rad), boundaries (sector limits),
        starts (start of the usable zone of each sector), centers (list of
        lists: centers[lp_idx][scenario_idx]), bar_width, ring_centers (same
        structure as centers, but tiling the whole wedge_span without gap),
        ring_width -- all the angles in RADIANS (polar ax.bar expects width in
        radians, not degrees).
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
    """Slope of the scale_UB scale beyond the Safe Limit: brings the Upper Safe
    Bound (ub_rel, in multiples of the Safe Limit) to UB_SCALED_REL. None if
    the row has no usable Upper Safe Bound (missing or <= Safe Limit) -- the
    sector then keeps the 1L scale."""
    if ub_rel is None or not np.isfinite(ub_rel) or ub_rel <= 1:
        return None
    return (UB_SCALED_REL - 1) / (ub_rel - 1)


def _scale_rel(rel, slope):
    """Scale specific to a sector (scale_UB, same logic as
    plot_overshoot.create_overshoot_safe_space_figure_by_region): piecewise
    linear, unchanged between 0 and the Safe Limit (1L), then with slope
    `slope` beyond (see _ub_slope). Identity if slope is None."""
    if rel is None or slope is None or rel <= 1:
        return rel
    return 1 + (rel - 1) * slope


def _format_tick(t):
    """"NL" label of a tick (integer or fractional)."""
    return f"{t:g}L"


def _radial_rotation_deg(theta_rad):
    """Rotation (screen degrees) for a text aligned along a radius (e.g. the
    "NL" ticks), always readable (never upside down). Assumes
    set_theta_zero_location('N') + set_theta_direction(-1)."""
    deg = (90.0 - np.degrees(theta_rad)) % 360
    if 90 < deg < 270:
        deg = (deg + 180) % 360
    return deg


def _tangential_rotation_deg(theta_rad):
    """Rotation (screen degrees) for a text aligned tangentially (along the
    circle, e.g. the LP names) -- orthogonal to _radial_rotation_deg."""
    deg = (-np.degrees(theta_rad)) % 360
    if 90 < deg < 270:
        deg = (deg + 180) % 360
    return deg


def _nice_radial_ticks(rel_cap, max_ticks=6):
    """Integer ticks (in multiples of the Safe Limit) from 1 to floor(rel_cap)
    (the actual radial ceiling being set separately, just above rel_cap -- see
    RADIAL_AXIS_MARGIN_FRAC -- a tick at rel_cap itself would make no sense if
    it is not an integer), subsampled if too many (same idea as
    plot_overshoot._set_break_axis_ticks). 1L is always included (Safe Limit
    circle)."""
    r_floor = max(1, int(np.floor(rel_cap)))
    ticks = list(range(1, r_floor + 1))
    if len(ticks) > max_ticks:
        step = max(1, int(np.ceil(r_floor / max_ticks)))
        ticks = list(range(step, r_floor + 1, step))
        if ticks[-1] != r_floor:
            ticks.append(r_floor)
    return sorted(set(ticks) | {1})


def _regular_radial_ticks(rel_top, max_ticks=5):
    """Ticks of a rescaled sector (scale_UB): 1L, then the multiples of a
    "round" step (1, 2, 2.5, 5 x 10^k) between 1L (excluded) and rel_top
    (included, in multiples of the Safe Limit, ACTUAL values -- before
    _scale_rel), with at most ~max_ticks ticks above 1L."""
    span = rel_top - 1
    if not np.isfinite(rel_top) or span <= 0:
        return [1]
    raw = span / max_ticks
    mag = 10 ** np.floor(np.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw * (1 - 1e-9))
    k_min = int(np.floor(1 / step + 1e-9)) + 1
    k_max = int(np.floor(rel_top / step + 1e-9))
    return [1] + [round(float(k * step), 10) for k in range(k_min, k_max + 1)]


# ============================================================================
# VALUE EXTRACTION, STACKING BY CATEGORY, ANNOTATION
# ============================================================================


def _stack_segments(payload, categories, threshold_ref, group_by_category,
                     show_categories, show_import_split, fallback_color):
    """Break down the CBA footprint of an (LP, scenario) into stackable
    segments, in units relative to threshold_ref (Safe Limit), ready to draw
    (color already resolved). to_r() must be applied by the caller to the
    cumulative BOUNDS (not to the height of the segment), the only way to stay
    correct on a non-linear scale (radial_power != 1).

    show_categories/show_import_split control the styling, independently:
      - both True: 1 segment per (category, dom/imp), colored by category
        (like create_stacked_bar_chart), hatched if imported.
      - show_categories=True, show_import_split=False: 1 segment per category
        (dom+imp merged), colored by category, never hatched.
      - show_categories=False, show_import_split=True: 2 segments (all
        domestic, all imported), colored with fallback_color (LP color), the
        2nd hatched.
      - both False: a single segment (total), colored with fallback_color --
        visually identical to a PBA bar.

    Returns:
        list of (rel_bottom, rel_top, color, is_import), in stacking order
        (dom then imp per group if group_by_category, otherwise all dom then
        all imp -- irrelevant if show_import_split=False).
    """
    if payload is None or threshold_ref is None:
        return []

    def value_for(key, category):
        return payload[key].get(category, 0) if key in payload else 0

    groups = list(enumerate(categories)) if show_categories else [(0, None)]

    def dom_imp(category):
        if category is not None:
            return value_for(DOMESTIC, category), value_for(IMPORTED, category)
        return (sum(value_for(DOMESTIC, c) for c in categories),
                sum(value_for(IMPORTED, c) for c in categories))

    entries = [(cat_idx, category, *dom_imp(category)) for cat_idx, category in groups]

    order = []  # (cat_idx, category, absolute_value, is_import)
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
    """Same text rule as create_synthesis_figure_by_lp: absolute value in bold
    black for the reference scenario, colored % variation (red = increase,
    green = decrease) for the others. PBA uses a full red/green (more saturated
    than the lightcoral/lightgreen of the CBA) to stay readable despite the
    light shade of the underlying PBA bar."""
    if is_reference or ref_val is None or ref_val <= 0:
        return f"{val:.0f}", "black"
    variation_pct = (val - ref_val) / ref_val * 100
    text = f"+{variation_pct:.0f}%" if variation_pct > 0 else f"{variation_pct:.0f}%"
    if is_pba:
        color = "#991616" if variation_pct > 0 else ("#237A23" if variation_pct < 0 else "black")
    else:
        color = "#F34E4B" if variation_pct > 0 else ("#7ADB67" if variation_pct < 0 else "black")
    return text, color


def _format_overshoot(rel):
    """Overshoot level of the Safe Limit, e.g. 2.23 -> "x2.2"."""
    return f"x{rel:.1f}"


def _bdf_column(bdf_df, indicator_name):
    """Column of the BdF sheet matching the indicator (named
    "<indicator> (<unit>)"), or None if missing."""
    for col in bdf_df.columns:
        if str(col).split(" (")[0].strip() == indicator_name:
            return col
    return None


# ============================================================================
# MAIN FIGURE
# ============================================================================


def create_radial_synthesis_figure(
    all_scenarios_data, thresholds_df, scenario_names,
    show_cba=True, show_pba=True,
    dls_df=None, show_dls=False,
    radial_power=0.5,
    show_categories=True, show_import_split=True, group_by_category=False,
    lp_colors=None,
    r_max=None,
    display_bounds=True,
    title=None,
    figsize=(16, 16),
    sharing_principle=None, shares_df=None,
    scale_UB=False, scale_diff=False,
    scenario_group=None,
    Bdf=None,
):
    """Radial wheel: 1 sector per LP, 1 bar per scenario, height = value /
    Safe Limit. The Safe Limit circle (green, historical threshold
    config.THRESHOLD_LB_ABS) is therefore at a constant radius = 1L over the
    whole figure; the Upper Safe Bound arc (red, config.THRESHOLD_UB_ABS) and
    the Lower Safe Bound arc (light green, config.THRESHOLD_LOWER_ABS, always
    below the Safe Limit) are specific to each sector (their ratio to the Safe
    Limit varies with the LP) -- these 3 names take the "raw" rows of
    thresholds.xlsx/sheet "All" ("Lower safe bound"/"Safe limit"/"Upper safe
    bound"). display_bounds=False hides Lower Safe Bound and Upper Safe Bound
    (only the Safe Limit circle remains). Outer ring: 1 sub-band per scenario;
    overshoot level of the Safe Limit for the reference scenario (e.g. "x2.2",
    legend "x2 = Overshoot"), % variation relative to it for the others.

    Args:
        all_scenarios_data: list of {lp_name: result of process_scenario},
            one item per scenario, in the order of scenario_names.
        show_cba, show_pba: which accountings to display (at least one of the
            two must be True):
            - show_cba alone: stackable bar, see show_categories/
              show_import_split.
            - show_pba alone: solid bar (a single scalar value exists in the
              source data, no breakdown by category/origin is ever possible),
              light shade of the LP color.
            - both True (default): side by side -- for each scenario, 2
              adjacent bars (CBA solid/stacked, PBA in light shade) and the
              outer ring follows the same split, 2 sub-bands per scenario.
              Title and legend adapt automatically.
            - only one of the two displayed ("solo" figure): the solid/light
              shades no longer distinguish CBA/PBA (redundant, there is only
              one) but the reference scenario (base_year, solid shade) from
              the other scenarios (light shade), for any cell where
              show_categories does not already take over (see below).
        show_categories: (CBA only) True = stacks 1 segment per consumption
            category, colored like create_stacked_bar_chart (the base_year/
            other scenarios distinction remains visible in the outer ring, but
            no longer in the segments themselves). False = no category
            distinction; the bar then takes the LP color, shaded by scenario
            as described above.
        show_import_split: (CBA only) True = distinguishes domestic (solid) /
            imported (hatched). False = domestic + imported merged into the
            same segment.
        group_by_category: see create_stacked_bar_chart -- only relevant if
            show_categories=True and show_import_split=True (dom/imp stacking
            order interleaved by category, or all dom then all imp).
        show_dls, dls_df: if show_dls=True, adds per sector the DLS (social
            floor), DLS lower bound and DLS upper bound arcs (shades of blue,
            light -> dark), when dls_df provides a value for this LP: national
            values read as is in the DLS sheet (see processing.lookup_dls).
            "DLS ..." rows if scenario_group is DLS_BASE_YEAR_GROUP
            ("Base_year"), "DLS 2050 ..." otherwise.
        scenario_group: displayed scenario group (config.get_scenario_folders),
            only used to choose the DLS rows (see show_dls) and the BdF file.
        radial_power: exponent of the transformation radius = (value/LB)**radial_power.
            1.0 = strictly linear scale. 0.5 (default) = square root. The
            smaller the value (e.g. 1/3), the more "dilated" the scale: the LPs
            below their limit (e.g. Blue water consumption, <1L) are stretched
            outwards and therefore more visible, while the LPs in large
            overshoot (e.g. GHG emissions) are all the more compressed -- same
            principle as sqrt, more pronounced. The Safe Limit circle stays at
            a constant radius (=1) whatever the value, since 1 raised to any
            power is always 1.
            Whatever its value, with scale_diff=True the radial ceiling is
            capped at SCALE_DIFF_CAP_REL (6L) -- see r_max.
        lp_colors: optional dict to override/extend LP_COLORS.
        r_max: optional ceiling of the radial axis, in multiples of the Safe
            Limit (same unit as the "NL" ticks); None = computed automatically
            from the data (values + displayed thresholds -- Upper/Lower Safe
            Bound if display_bounds=True, DLS if show_dls=True), capped at
            SCALE_DIFF_CAP_REL if scale_diff=True. With scale_diff=True, any
            bar exceeding the ceiling is truncated at the outer circle, with
            its actual level (e.g. "16L") written at its top; the threshold
            arcs beyond the ceiling are not drawn.
        display_bounds: True (default) displays the Lower Safe Bound and Upper
            Safe Bound arcs in addition to the Safe Limit circle (always
            displayed). False hides them (and no longer counts them in the
            computation of r_max).
        title: figure title (e.g. "Planetary boundaries assessment of French
            NZE Scenarios").
        sharing_principle: None (default) = thresholds read as is in
            thresholds_df ("Equality (Lower bound)"/"Equality"/"Equality (Upper
            bound)" rows). "EPC": France budget = world budget ("Lower safe
            bound"/"Safe limit"/"Upper safe bound") x share of the "FR" row of
            budget_shares.xlsx, column "UN – EPC 2050" ("UN – EPC 2019-2100"
            for GHG emissions) -- see config.radial_epc_pattern. "CTR": same
            with the "CTR share ref. ..." column.
        shares_df: shares of budget_shares.xlsx (io.load_budget_shares_df),
            used for "EPC"/"CTR"; loaded automatically if None.
        scale_UB: if False (default), common scale for all the sectors, in
            multiples of the Safe Limit (1L, 2L, 3L...). If True (same principle
            as plot_overshoot.create_overshoot_safe_space_figure_by_region), the
            scale of each sector is specific to its LP, piecewise linear (before
            radial_power): unchanged between 0 and the Safe Limit (1L), then
            stretched or compressed so that the Upper Safe Bound falls exactly
            on UB_SCALED_REL (2L); beyond, the slope of the [Safe Limit, UB]
            segment is extended. The figure then shows two continuous
            concentric circles: Safe Limit (green) and Upper Safe Bound (red,
            if display_bounds=True). Bars, Lower Safe Bound and DLS follow this
            scale. The "NL" ticks beyond 1L, which no longer have a common
            meaning, are hidden. A sector without Upper Safe Bound (or with UB
            <= Safe Limit) keeps the 1L scale.
        scale_diff: False (default) = classic display, common scale up to the
            highest point of the figure. True = radial ceiling capped at
            SCALE_DIFF_CAP_REL (6L): the bars exceeding it go up to the outer
            circle and their actual level (e.g. "16L") is written just below
            (see r_max). Incompatible with scale_UB.
        Bdf: None (default) = classic multi-scenario figure. Otherwise, name of
            a sheet (e.g. "décile") of the BdF file of the scenario_group group
            (io.bdf_file: impacts_facteurs_agrégé_{scenario_group}*.xlsx): the
            scenarios are replaced by the aggregate value ("Average" row, in
            1st position, solid shade) then the household categories of the 1st
            column of this sheet (light shade), 1 solid bar each, of height =
            pressure of the category / (national Safe Limit x weight) -- each
            category gets the "weight" share of the national budget. Upper/Lower
            Safe Bound and DLS being weighted by the same weight, their ratio
            to the Safe Limit is unchanged: the arcs remain common to the
            whole sector. The outer ring displays for the Total its overshoot
            level of the Safe Limit (e.g. "x2.2", legend "x2 = Overshoot") and,
            for each category, the relative variation of its overshoot (bar
            height) relative to that of the Total (e.g. +20%). show_cba/
            show_pba, show_categories/show_import_split and the sorting of the
            scenarios are ignored; all_scenarios_data only sets the list/order
            of the LPs (reference scenario).
            If the folder of the scenario_group group contains 2 scenarios
            including a base year (e.g. "S1+" = 2019_FR + S1_2050, see
            io.bdf_comparison_file), the bars (file of the group) remain
            unchanged and the gap to the base year (Base_year file) is added on
            each of them: white bar tip with a thin black outline up to the
            base-year value if it is higher, outline only on the top of the bar
            if it is lower (legend "Base year"). show_dls then draws the central
            base-year DLS ("DLS Base year") and the DLS 2050 (darker blue)
            instead of the 3 DLS levels. False is equivalent to None.

    Returns:
        (fig, ax)
    """
    if not show_cba and not show_pba:
        raise ValueError("show_cba and show_pba cannot both be False.")
    if scale_UB and scale_diff:
        raise ValueError("scale_UB and scale_diff cannot both be True.")
    is_bdf = Bdf not in (None, False)
    if is_bdf:
        show_cba, show_pba = True, False
    combined = show_cba and show_pba

    reference_idx = config.REFERENCE_SCENARIO_IDX
    if not is_bdf:
        # The reference scenario (base_year) keeps its position, followed by the
        # trend scenario (config.TREND_SCENARIO_MARKER), then the other
        # scenarios by decreasing mean variation of the footprint vs the
        # reference (see processing.mean_variation): the smallest decrease /
        # largest increase right next to the trend scenario, the largest
        # decrease the furthest away.
        value_keys = (["cba"] if show_cba else []) + (["pba"] if show_pba else [])
        display_order = processing.sort_scenarios_by_reduction(all_scenarios_data, scenario_names, reference_idx, value_keys)
        scenario_names = [scenario_names[i] for i in display_order]
        all_scenarios_data = [all_scenarios_data[i] for i in display_order]

    first_scenario_data = all_scenarios_data[reference_idx]
    indicators = list(first_scenario_data.keys())

    comp_df = None
    if is_bdf:
        # The "scenarios" become the aggregate value ("Average" row, in 1st
        # position = reference) followed by the household categories.
        bdf_df = io.load_bdf(Bdf, scenario_group)
        bdf_df = pd.concat([bdf_df.loc[[config.BDF_TOTAL_ROW]],
                            bdf_df.drop(index=config.BDF_TOTAL_ROW)])
        bdf_weights = bdf_df[config.BDF_WEIGHT_COL].astype(float).tolist()
        scenario_names = list(bdf_df.index)
        reference_idx = 0
        # Group of 2 scenarios (e.g. S1+ = 2019_FR + S1_2050): same categories
        # read in the BdF file of the base year, to draw the difference.
        comparison_path = io.bdf_comparison_file(scenario_group)
        comp_df = None
        if comparison_path is not None:
            comp_df = io.load_bdf(Bdf, path=comparison_path)
            missing = [c for c in scenario_names if c not in comp_df.index]
            if missing:
                raise ValueError(f"Categories missing from {comparison_path.name}: {missing}")
            comp_df = comp_df.loc[scenario_names]
            comp_weights = comp_df[config.BDF_WEIGHT_COL].astype(float).tolist()
    bdf_compare = is_bdf and comp_df is not None

    # DLS levels drawn: (key, row of the DLS sheet, color, label, scope).
    # BdF comparison: central base-year DLS and DLS 2050; otherwise the 3 levels
    # of config.DLS_LEVELS, "DLS ..." rows (base-year group) or "DLS 2050 ...".
    # 5th item: bars covered by the arc ("all" = whole sector, "ref" = reference
    # scenario only, "others" = all the other scenarios).
    if bdf_compare:
        dls_levels = [("dls_base", "DLS", config.DLS_COLOR, "DLS Base year", "all"),
                      ("dls_2050", "DLS 2050", config.DLS_2050_COLOR, "DLS 2050", "all")]
    elif not is_bdf and scenario_group != config.DLS_BASE_YEAR_GROUP:
        # 2050 multi-scenario: base-year DLS on the Base year bar only, DLS 2050
        # on the other scenarios.
        dls_levels = [("dls_base", "DLS", config.DLS_COLOR, "DLS Base year", "ref"),
                      ("dls_2050", "DLS 2050", config.DLS_2050_COLOR, "DLS 2050", "others")]
    else:
        is_base_year = scenario_group == config.DLS_BASE_YEAR_GROUP
        dls_prefix = "DLS" if is_base_year else "DLS 2050"
        # Legend label specifying the DLS horizon ("DLS Base year ..." /
        # "DLS 2050 ...") instead of the generic "DLS" prefix.
        label_prefix = "DLS Base year" if is_base_year else "DLS 2050"
        dls_levels = [(key, dls_prefix + suffix, color, label_prefix + label[len("DLS"):], "all")
                      for key, suffix, color, label in config.DLS_LEVELS]

    n_lp = len(indicators)
    n_scenarios = len(scenario_names)
    # In combined mode, each scenario takes 2 adjacent slots (CBA then PBA):
    # the geometry is built for 2×n_scenarios virtual "slots", and the actual
    # scenario is found via slot_idx // 2.
    n_slots = n_scenarios * 2 if combined else n_scenarios

    palette = dict(LP_COLORS)
    if lp_colors:
        palette.update(lp_colors)

    def to_r(rel):
        return float(rel) ** radial_power

    geo = _wedge_geometry(n_lp, n_slots)

    # FIRST PASS: values, thresholds, unit and categories by LP (and maximum
    # radius needed to encompass values + displayed UB/DLS thresholds)
    lp_info = []
    legend_categories = set()
    max_rel = 1.0
    is_share = sharing_principle is not None
    if is_share:
        if sharing_principle not in config.BUDGET_SHARE_COLUMN_PATTERNS:
            raise ValueError(f"Unknown sharing_principle: {sharing_principle!r} "
                             f"(expected: None, {', '.join(config.BUDGET_SHARE_COLUMN_PATTERNS)})")
        if shares_df is None:
            shares_df = io.load_budget_shares_df()
        if sharing_principle == "EPC":
            for pattern in {config.RADIAL_EPC_PATTERN, *config.RADIAL_EPC_PATTERN_BY_INDICATOR.values()}:
                if not any(re.search(pattern, str(c), flags=re.IGNORECASE) for c in shares_df.columns):
                    raise ValueError(
                        f"No column of shares_df matches {pattern!r}: regenerate "
                        f"budget_shares.xlsx (\"UN – EPC\" columns)."
                    )

    def threshold(threshold_row, threshold_kind, indicator_name):
        if is_share:
            # France budget = world budget x share of the "FR" row: UN EPC
            # ("UN – EPC 2050", or 2019-2100 for GHG, see config.radial_epc_pattern)
            # or reference CTR share ("CTR share ref. ...").
            column_pattern = (config.radial_epc_pattern(indicator_name)
                              if sharing_principle == "EPC" else None)
            return processing.compute_region_budget(
                thresholds_df, shares_df, indicator_name, config.RADIAL_EPC_REGION_CODE, sharing_principle,
                threshold_kind=threshold_kind, column_pattern=column_pattern,
            )
        return processing.lookup_threshold(thresholds_df, threshold_row, indicator_name)

    for indicator_name in indicators:
        threshold_lb = threshold(config.THRESHOLD_LB_ABS, "lb", indicator_name)
        threshold_ub = threshold(config.THRESHOLD_UB_ABS, "ub", indicator_name)
        threshold_lower = threshold(config.THRESHOLD_LOWER_ABS, "lower", indicator_name)
        unit_text = processing.lookup_first_text(thresholds_df, config.UNIT_ROW_ABS, indicator_name)
        # DLS (and its bounds), by key of config.DLS_LEVELS; None if missing for this LP
        dls_values = {key: None for key, *_ in dls_levels}
        if show_dls and dls_df is not None:
            for key, dls_row, *_ in dls_levels:
                dls_values[key] = processing.lookup_dls(dls_df, dls_row, indicator_name)

        categories = (first_scenario_data.get(indicator_name) or {}).get("categories", [])
        legend_categories.update(categories)

        if is_bdf:
            # 1 value per household category (Total first).
            bdf_col = _bdf_column(bdf_df, indicator_name)
            values = [float(v) for v in bdf_df[bdf_col]] if bdf_col is not None else [None] * n_scenarios
            # Thresholds specific to each category: national threshold x weight.
            weights = bdf_weights
            scale_values = values
            if bdf_compare:
                comp_col = _bdf_column(comp_df, indicator_name)
                comp_values = ([float(v) for v in comp_df[comp_col]] if comp_col is not None
                               else [None] * n_scenarios)
        else:
            cba_values = [processing.extract_value(sd.get(indicator_name), "cba") for sd in all_scenarios_data]
            pba_values = [processing.extract_value(sd.get(indicator_name), "pba") for sd in all_scenarios_data]
            if combined:
                values = []
                for cba_val, pba_val in zip(cba_values, pba_values):
                    values.append(cba_val)
                    values.append(pba_val)
            else:
                values = cba_values if show_cba else pba_values
            weights = [1.0] * n_slots
            # The radial scale ("NL" ticks, ceiling) is always based on the
            # CBA/PBA max, even in solo mode (show_cba alone or show_pba alone):
            # a PBA-only figure must have the same ticks as a CBA-only or
            # combined figure for the same sector, not ticks tightened on its
            # displayed values only.
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
            if bdf_compare:
                # Same normalization as the bars (national threshold x weight
                # of the category, read in the comparison file).
                comp_rel_values = [(v / (threshold_ref * w)) if v is not None else None
                                   for v, w in zip(comp_values, comp_weights)]
                scale_rel_values = scale_rel_values + comp_rel_values
        else:
            rel_values = [None] * n_slots
            scale_rel_values = []
            ub_rel = None
            lb_rel = None
            lower_rel = None
            dls_rels = {key: None for key in dls_values}
        if not bdf_compare or threshold_ref is None:
            comp_rel_values = [None] * n_scenarios

        # scale_UB: slope specific to the sector (None = unchanged 1L scale).
        # The relative values remain stored without rescaling; _scale_rel is
        # applied when drawing (see r_of).
        slope = _ub_slope(ub_rel) if scale_UB else None

        bound_rels = [ub_rel, lower_rel] if display_bounds else []
        for rel in scale_rel_values + [lb_rel, *dls_rels.values()] + bound_rels:
            if rel is not None and np.isfinite(rel):
                max_rel = max(max_rel, _scale_rel(rel, slope))

        lp_info.append(dict(
            name=indicator_name, unit=unit_text, threshold_ref=threshold_ref, categories=categories,
            values=values, rel_values=rel_values, ub_rel=ub_rel, lb_rel=lb_rel, lower_rel=lower_rel,
            dls_rels=dls_rels, slope=slope, ub_slope=slope, comp_rel_values=comp_rel_values,
            color=palette.get(indicator_name, _FALLBACK_LP_COLOR),
        ))

    if r_max is not None:
        rel_cap_input = r_max
    elif scale_diff:
        rel_cap_input = min(max_rel, SCALE_DIFF_CAP_REL)
    else:
        rel_cap_input = max_rel

    def is_clipped(info, rel):
        """True if the relative value exceeds the radial ceiling (rel_cap)."""
        return rel is not None and _scale_rel(rel, info["slope"]) > rel_cap

    def r_of(info, rel):
        """Drawn radius of a relative value of the sector info: scale_UB
        rescaling (identity otherwise), truncated at the radial ceiling (outer
        circle), then radial_power."""
        return to_r(min(_scale_rel(rel, info["slope"]), rel_cap))

    # The radial ceiling is set just above (RADIAL_AXIS_MARGIN_FRAC) the
    # highest relative value (bar or displayed UB/LB/DLS threshold), so that the
    # outer ring starts very close to the actual top -- not at the rounding up
    # to the next whole L, which could leave up to 1L of dead space. The "NL"
    # ticks remain integer multiples (up to the last one <= this actual top).
    # scale_UB: only the 1L tick remains common to all the sectors; each sector
    # also gets its own regular ticks (actual values), up to the top of its
    # scale (rel_cap_input brought back to an actual value by the inverse of
    # _scale_rel).
    rel_ticks = [1] if scale_UB else _nice_radial_ticks(rel_cap_input)
    for info in lp_info:
        if not scale_UB:
            info["rel_ticks"] = rel_ticks
            continue
        slope = info["slope"]
        rel_top = rel_cap_input if (slope is None or rel_cap_input <= 1) else 1 + (rel_cap_input - 1) / slope
        info["rel_ticks"] = _regular_radial_ticks(rel_top)
    rel_cap = rel_cap_input * (1 + RADIAL_AXIS_MARGIN_FRAC)
    r_max_plot = to_r(rel_cap)
    ring_thickness = r_max_plot * 0.11
    r_ring_start = r_max_plot
    r_ring_end = r_ring_start + ring_thickness
    r_label = r_max_plot * 0.78
    # Radius of the scenario names recalled above the cells of the outer ring
    # (SCENARIO_LABEL_INDICATORS only) -- just outside the ring, see
    # _nice_radial_ticks/set_rlim below.
    r_scenario_label = r_ring_end + ring_thickness * 0.7
    has_scenario_labels = any(name in SCENARIO_LABEL_INDICATORS for name in indicators)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="polar")
    # Default matplotlib margins are very generous for a single subplot (~12% on
    # each side): they are tightened so that the circle fills the figure, with
    # just the room needed above for the title and below for the legend
    # (positioned right after, flush with these new bounds).
    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.058)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    # Background grid: concentric circles + spokes at the sector limits, with
    # "NL" ticks repeated at each of the n_lp limits.
    # scale_UB: ticks specific to each sector -> arcs limited to the sector
    # (instead of full circles), radius after _scale_rel.
    theta_full = np.linspace(0, 2 * np.pi, 400)
    if not scale_UB:
        for t in rel_ticks:
            if t == 1:
                continue  # drawn separately, in bold, as the Safe Limit circle
            ax.plot(theta_full, np.full_like(theta_full, to_r(t)), color="#e0e0e0", linewidth=0.7, zorder=1)

    for lp_idx, b in enumerate(geo["boundaries"]):
        info = lp_info[lp_idx]
        ax.plot([b, b], [0, r_ring_end], color="#bdbdbd", linewidth=0.8, zorder=1)
        offset = geo["wedge_span"] * 0.02
        if scale_UB:
            theta_sector = np.linspace(b, b + geo["wedge_span"], 60)
            for t in info["rel_ticks"]:
                if t == 1:
                    continue
                ax.plot(theta_sector, np.full_like(theta_sector, r_of(info, t)), color="#e0e0e0",
                        linewidth=0.7, zorder=1)
        for t in info["rel_ticks"]:
            ax.text(b + offset, r_of(info, t), _format_tick(t), ha="left", va="center", fontsize=12,
                     color="#555555", rotation=_radial_rotation_deg(b), rotation_mode="anchor",
                     bbox=dict(facecolor="white", edgecolor="white", alpha=0.7, pad=0.5), zorder=3)

    # SECOND PASS: bars, outer ring, UB/DLS arcs, LP name
    for lp_idx, info in enumerate(lp_info):
        centers = geo["centers"][lp_idx]
        bar_width = geo["bar_width"]
        ring_centers = geo["ring_centers"][lp_idx]

        def _bar_hatch(is_import=False):
            # "/" hatching for the imported part only (the reference scenario
            # is no longer hatched, see _solo_color).
            return "/" if is_import else None

        # PBA: always in a lighter shade of the LP color.
        pba_color = colors.get_light_shade(info["color"], 1.5)

        def _solo_color(scenario_idx):
            # Outside combined mode, CBA and PBA now share a single color per LP
            # (that of the PBA): the two shades no longer distinguish CBA/PBA
            # (only one is displayed) but the reference scenario (base_year,
            # solid shade -- the former "CBA" color) from the other scenarios
            # (light shade), instead of the "xx" hatching used before. BdF: the
            # aggregate value (Total) plays the role of the reference scenario.
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
            # 2 adjacent bars per scenario -- CBA (stacked like _draw_cba_bar)
            # then PBA (always solid, light shade).
            for scenario_idx, theta in enumerate(centers[0::2]):
                _draw_cba_bar(theta, scenario_idx)
            pba_centers = centers[1::2]
            pba_heights = [r_of(info, rel) if rel is not None else 0.0 for rel in info["rel_values"][1::2]]
            for scenario_idx, (theta, height) in enumerate(zip(pba_centers, pba_heights)):
                ax.bar([theta], [height], width=bar_width, bottom=0.0,
                       color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        elif show_cba and not is_bdf:
            # Detailed footprint available (domestic/imported by category): bar
            # stacked according to show_categories/show_import_split (both
            # False -> 1 solid segment, colored by _solo_color). to_r() is
            # applied to the cumulative BOUNDS (not to the height of the
            # segment), the only correct way to stay consistent with the LB
            # circle/UB arc on a non-linear scale (radial_power != 1).
            for scenario_idx, theta in enumerate(centers):
                _draw_cba_bar(theta, scenario_idx)
        else:
            # PBA alone (or BdF): a single scalar value per (LP, scenario) in
            # the source data -> no breakdown possible, solid bar.
            heights = [r_of(info, rel) if rel is not None else 0.0 for rel in info["rel_values"]]
            for scenario_idx, (theta, height) in enumerate(zip(centers, heights)):
                ax.bar([theta], [height], width=bar_width, bottom=0.0,
                       color=_solo_color(scenario_idx), edgecolor="white", linewidth=0.6, zorder=5)

        if bdf_compare:
            # Difference with the base year, without touching the bars: if the
            # base year is higher, white bar tip with a thick black outline put
            # on the bar; if it is lower, outline only (no fill) on the top of
            # the bar, which remains visible.
            for theta, rel, comp_rel in zip(centers, info["rel_values"], info["comp_rel_values"]):
                if rel is None or comp_rel is None or np.isclose(rel, comp_rel):
                    continue
                r_bar, r_comp = r_of(info, rel), r_of(info, comp_rel)
                if r_comp == r_bar:
                    continue  # both beyond the radial ceiling
                ax.bar([theta], [abs(r_comp - r_bar)], width=bar_width, bottom=min(r_bar, r_comp),
                       facecolor="white" if r_comp > r_bar else "none",
                       edgecolor=COMPARISON_EDGE_COLOR, linewidth=COMPARISON_EDGE_WIDTH, zorder=6)

        # Outer ring: full tiling of the sector (ring_centers/ring_width,
        # without the gap_frac of the bars) to leave no gap between scenarios or
        # between sectors. In combined mode, the PBA half takes the light shade
        # of the matching bar; otherwise each cell follows the color of the bar
        # of the matching scenario (_solo_color).
        if combined:
            ax.bar(ring_centers[0::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=info["color"], edgecolor="white", linewidth=0.6, zorder=5)
            ax.bar(ring_centers[1::2], [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=pba_color, edgecolor="white", linewidth=0.6, zorder=5)
        else:
            ring_colors = [_solo_color(scenario_idx) for scenario_idx in range(n_scenarios)]
            ax.bar(ring_centers, [ring_thickness] * n_scenarios, width=geo["ring_width"], bottom=r_ring_start,
                   color=ring_colors, edgecolor="white", linewidth=0.6, zorder=5)

        # Ring text: value/reference scenario depending on the slot (CBA or
        # PBA) in combined mode, on the only displayed accounting otherwise.
        if combined:
            ref_cba, ref_pba = info["values"][reference_idx * 2], info["values"][reference_idx * 2 + 1]
            ref_by_slot = [ref_cba if i % 2 == 0 else ref_pba for i in range(n_slots)]
            is_ref_by_slot = [(i // 2) == reference_idx for i in range(n_slots)]
            is_pba_by_slot = [i % 2 == 1 for i in range(n_slots)]
        else:
            ref_by_slot = [info["values"][reference_idx]] * n_slots
            is_ref_by_slot = [i == reference_idx for i in range(n_slots)]
            # A single accounting displayed: the color of the variations always
            # follows the PBA style (saturated red/green), whether it is the CBA
            # or the PBA that is displayed alone.
            is_pba_by_slot = [True] * n_slots

        r_ring_mid = (r_ring_start + r_ring_end) / 2
        for i, theta in enumerate(ring_centers):
            val = info["values"][i]
            if val is None:
                continue
            if is_ref_by_slot[i]:
                # Reference scenario (BdF: Total), CBA and PBA in combined mode:
                # overshoot level of the Safe Limit (1L), e.g. "x2.2", rather
                # than the absolute value (legend "x2 = Overshoot").
                rel = info["rel_values"][i]
                if rel is None:
                    continue
                text, color = _format_overshoot(rel), "black"
            elif is_bdf:
                # Relative variation of the overshoot (pressure / weighted
                # threshold) of the category compared with that of the
                # aggregate value.
                text, color = _annotate(info["rel_values"][i], info["rel_values"][reference_idx], False, True)
            else:
                text, color = _annotate(val, ref_by_slot[i], is_ref_by_slot[i], is_pba_by_slot[i])
            ax.text(theta, r_ring_mid, text, ha="center", va="center",
                    fontsize=14.5, fontweight="bold", color=color, zorder=9)

        # Scenario name recalled above the cell(s) of the outer ring -- only
        # for the 2 opposite LPs on the wheel (replaces the global "Scenario
        # order: ..." subtitle). In combined mode, a scenario takes 2 adjacent
        # cells (CBA then PBA): the name is centered above both.
        if info["name"] in SCENARIO_LABEL_INDICATORS:
            for scenario_idx in range(n_scenarios):
                if combined:
                    theta_label = (ring_centers[scenario_idx * 2] + ring_centers[scenario_idx * 2 + 1]) / 2
                else:
                    theta_label = ring_centers[scenario_idx]
                ax.text(theta_label, r_scenario_label, scenario_names[scenario_idx], ha="center", va="center",
                        fontsize=12, fontweight="bold", color="#333333",
                        rotation=_tangential_rotation_deg(theta_label), rotation_mode="anchor", zorder=9)

        # Actual level of the bars truncated at the radial ceiling (e.g.
        # "16L"), written in the middle of their top, just below the outer circle.
        for theta, rel in zip(centers, info["rel_values"]):
            if is_clipped(info, rel):
                ax.text(theta, r_max_plot * 0.97, _format_tick(round(rel)), ha="center", va="center",
                        fontsize=12, fontweight="bold", color="black",
                        rotation=_tangential_rotation_deg(theta), rotation_mode="anchor", zorder=9,
                        path_effects=[pe.withStroke(linewidth=3, foreground="white")])

        if (display_bounds and info["ub_rel"] is not None and info["ub_slope"] is None
                and not is_clipped(info, info["ub_rel"])):
            # Rescaled sector (scale_UB): its Upper Safe Bound is drawn further
            # down, with the continuous circle at UB_SCALED_REL.
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, r_of(info, info["ub_rel"])), color=UB_COLOR,
                    linewidth=3.0, solid_capstyle="butt", zorder=8)

        if display_bounds and info["lower_rel"] is not None:
            theta_arc = np.linspace(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"], 40)
            ax.plot(theta_arc, np.full_like(theta_arc, r_of(info, info["lower_rel"])), color=LOWER_COLOR,
                    linewidth=3.0, solid_capstyle="butt", zorder=8)

        # DLS arcs: the whole usable sector, except in combined mode where they
        # only cover the CBA bars (the DLS is a consumption floor, irrelevant
        # for the adjacent PBA bar).
        # Scope "ref"/"others": arc limited to the bars of the reference
        # scenario or of the other scenarios (CBA bars only in combined mode).
        cba_centers = centers[0::2] if combined else centers
        for key, _, dls_color, _, scope in dls_levels:
            dls_rel = info["dls_rels"][key]
            if dls_rel is None or is_clipped(info, dls_rel):
                continue
            if scope == "all" and not combined:
                dls_spans = [(geo["starts"][lp_idx], geo["starts"][lp_idx] + geo["usable_span"])]
            elif scope == "ref":
                # Widened arc: from the edge of the sector (or of the neighboring
                # bar) up to the next bar, on both sides of the Base year bar.
                slot = reference_idx * 2 if combined else reference_idx
                sector_start = geo["boundaries"][lp_idx]
                theta_start = (centers[slot - 1] + bar_width / 2) if slot > 0 else sector_start
                theta_end = ((centers[slot + 1] - bar_width / 2) if slot + 1 < len(centers)
                             else sector_start + geo["wedge_span"])
                dls_spans = [(theta_start, theta_end)]
            else:
                dls_spans = [(theta - bar_width / 2, theta + bar_width / 2)
                             for scenario_idx, theta in enumerate(cba_centers)
                             if scope == "all" or (scenario_idx == reference_idx) == (scope == "ref")]
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

    # Safe Limit "1L" circle: constant radius = 1 (fixed point of to_r, whatever
    # radial_power), drawn above the bars.
    ax.plot(theta_full, np.full_like(theta_full, to_r(1.0)), color=LB_COLOR, linewidth=3.2, zorder=8)
    for lp_idx, info in enumerate(lp_info):
        theta_arc = np.linspace(geo["boundaries"][lp_idx], geo["boundaries"][lp_idx] + geo["wedge_span"], 40)
        # scale_UB: second concentric circle, Upper Safe Bound brought to
        # UB_SCALED_REL in each rescaled sector -- arcs tiling the whole
        # wedge_span to form a continuous circle.
        if display_bounds and info["ub_slope"] is not None:
            ax.plot(theta_arc, np.full_like(theta_arc, to_r(UB_SCALED_REL)), color=UB_COLOR,
                    linewidth=3.2, zorder=8)

    r_axis_max = (r_scenario_label + ring_thickness * 0.6) if has_scenario_labels else (r_ring_end * 1.02)
    ax.set_rlim(0, r_axis_max)

    # Legend: threshold lines common to both figures, then the categories if
    # displayed (CBA with show_categories=True -- explains the styling of the
    # stacked bars) and/or the "Import" entry alone if the dom/imp distinction
    # is displayed without the categories.
    legend_handles = [Line2D([0], [0], color=LB_COLOR, linewidth=3, label="Safe Limit")]
    if display_bounds:
        legend_handles = [
            Line2D([0], [0], color=LOWER_COLOR, linewidth=3, label="Lower Safe Bound"),
            *legend_handles,
            Line2D([0], [0], color=UB_COLOR, linewidth=3, label="Upper Safe Bound"),
        ]
    if show_dls:
        for key, _, dls_color, dls_label, _ in dls_levels:
            if any(info["dls_rels"][key] is not None for info in lp_info):
                legend_handles.append(Line2D([0], [0], color=dls_color, linestyle="--", linewidth=3, label=dls_label))

    if is_bdf:
        pass  # solid bars, one color per LP: nothing to explain beyond the thresholds
    elif show_cba and show_categories:
        cat_handles, cat_labels = _build_category_legend(colors.sort_categories(list(legend_categories)))
        if not show_import_split:
            cat_handles, cat_labels = cat_handles[:-1], cat_labels[:-1]  # drop "Import": no hatching drawn
        for h, lbl in zip(cat_handles, cat_labels):
            h.set_label(lbl)
        legend_handles += cat_handles
    elif show_cba and show_import_split:
        legend_handles.append(Patch(facecolor="white", edgecolor="black", hatch="///", linewidth=1, label="Import"))

    if combined:
        # Generic marker (neutral grey) for the light shade of the PBA, valid
        # for all the LPs: the two shades distinguish CBA/PBA here.
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Consumption-based"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Production-based"))
    elif is_bdf:
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Average household"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Income decile groups"))
        if bdf_compare:
            legend_handles.append(Patch(facecolor="white", edgecolor=COMPARISON_EDGE_COLOR,
                                        linewidth=COMPARISON_EDGE_WIDTH, label="Base year"))
    else:
        # A single accounting displayed: the two shades instead distinguish the
        # reference scenario from the others (see _solo_color).
        legend_handles.append(Patch(facecolor="#888888", edgecolor="#333333", label="Base year"))
        legend_handles.append(Patch(facecolor=colors.get_light_shade("#888888", 1.5), edgecolor="#333333", label="Other scenarios"))

    # Cell of the reference scenario (BdF: Total) in the outer ring: overshoot
    # level of the Safe Limit (x2.2...)
    legend_handles.append(Line2D([0], [0], linestyle="none", marker="$x2$", markersize=18,
                                 color="black", label="Overshoot of Safe Limit (1L)"))

    # Relative variation of the footprints
    if is_bdf:
        legend_handles.append(Line2D([0], [0], linestyle="none", marker=r"$+10\%$", markersize=32,
                                     color="#F34E4B" if combined else "#991616", label="Per capita footprint vs. Average"))
    else:
        legend_handles.append(Line2D([0], [0], linestyle="none", marker=r"$+10\%$", markersize=32,
                                     color="#F34E4B" if combined else "#991616", label="Footprint change vs. Baseline"))

    # matplotlib fills the legend column by column: the grid is completed with
    # empty cells so that the last entry (footprint variation) falls at the
    # bottom right, hence on the 2nd row (at least 2 rows).
    n_rows = max(2, -(-len(legend_handles) // 5))
    n_cols = -(-len(legend_handles) // n_rows)
    n_pad = n_rows * n_cols - len(legend_handles)
    legend_handles = (legend_handles[:-1]
                      + [Line2D([0], [0], linestyle="none", label=" ") for _ in range(n_pad)]
                      + legend_handles[-1:])

    fig.legend(handles=legend_handles, loc="lower center", ncol=n_cols,
               bbox_to_anchor=(0.5, 0.015), frameon=False, fontsize=15)

    fig.suptitle(title, fontsize=18, fontweight="bold", y=0.975)

    return fig, ax
