"""
Processing of the data loaded by io.py: filtering/weighting by characterization
factor, aggregation by indicator, computation of the footprints in absolute
values (process_scenario, broken down by consumption category, and
process_scenario_absolute, total only) and per capita CBA/PBA
(process_scenario_per_capita, population from the "Population" sheet of
budget_shares.xlsx).

Also contains the descent from the world budget to the budget of a country
through EPC/CTR regional shares (compute_region_budget), and the per-capita
sensitivity of the World budget (compute_world_period_budget).
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import colors, config, io

# Keys of the footprint payloads ({"domestic": Series, "imported": Series,
# "pba": float, "categories": [...]}) shared by all the figures.
DOMESTIC = "domestic"
IMPORTED = "imported"

# ============================================================================
# CHARACTERIZATION FACTORS: indicator <-> LP, stressor -> factor
# ============================================================================


def get_unique_indicators(cf_df):
    """Return {indicator: [LP1, LP2, ...]} from the characterization factors."""
    indicator_to_lp = {}
    for _, row in cf_df.iterrows():
        sp = row[config.CF_INDICATOR_COL]
        lp = row[config.CF_LP_COL]
        indicator_to_lp.setdefault(sp, [])
        if lp not in indicator_to_lp[sp]:
            indicator_to_lp[sp].append(lp)
    return indicator_to_lp


def build_extension_factor_map(cf_df, indicator_name, lp_name=None):
    """Build {exiobase_stressor: characterization_factor} for a indicator.

    If lp_name is given, also restricts to the matching LP. Both modes are
    actually used (not just for convenience): process_scenario() keeps all the
    stressors of the indicator whatever the current LP (each d_cba file only
    contains the stressors of its own LP anyway), whereas
    _scenario_footprint_totals() explicitly restricts to the LP -- both give
    the same numerical result, but do not merge the two modes without
    checking: it is not just a matter of style.
    """
    mask = cf_df[config.CF_INDICATOR_COL] == indicator_name
    if lp_name is not None:
        mask &= cf_df[config.CF_LP_COL] == lp_name
    subset = cf_df[mask]
    return dict(zip(subset[config.CF_STRESSOR_COL].values, subset[config.CF_FACTOR_COL].values))


# ============================================================================
# FILTERING + WEIGHTING BY CHARACTERIZATION FACTOR
# ============================================================================
# Primitive shared by all footprint computations: keep only the rows
# (exiobase stressors) present in extension_factor_map, and multiply them by
# their factor. filter_and_weight() does the work; the two wrappers only apply
# a different reduction (vector by category, or scalar).


def filter_and_weight(df, extension_factor_map):
    """Filter the rows of `df` whose index (1st level if MultiIndex) is a key
    of `extension_factor_map`, and multiply them by the matching factor.

    Returns:
        pd.DataFrame: the retained weighted rows (may be empty).
    """
    weighted_rows = [
        row * extension_factor_map[idx[0] if isinstance(idx, tuple) else idx]
        for idx, row in df.iterrows()
        if (idx[0] if isinstance(idx, tuple) else idx) in extension_factor_map
    ]
    return pd.DataFrame(weighted_rows) if weighted_rows else pd.DataFrame()


def filter_and_weight_vector(df, extension_factor_map):
    """Like filter_and_weight(), reduced by summing the rows -> a Series (1
    value per column of `df`, hence per category/sector/region depending on the case)."""
    weighted = filter_and_weight(df, extension_factor_map)
    if weighted.empty:
        return pd.Series(dtype=float)
    summed = weighted.sum(axis=0)
    if isinstance(summed.index, pd.MultiIndex):
        summed.index = summed.index.get_level_values(-1)
    return summed


def filter_and_weight_scalar(df, extension_factor_map):
    """Like filter_and_weight(), reduced by summing everything -> a single total (float)."""
    weighted = filter_and_weight(df, extension_factor_map)
    return float(weighted.to_numpy().sum()) if not weighted.empty else 0.0


# ============================================================================
# SAFE LOOKUPS IN thresholds_df (avoids repeating the same try/except everywhere)
# ============================================================================


def lookup_cell(thresholds_df, row_name, column_name, require_positive=False):
    """Get thresholds_df.loc[row_name, column_name] as a float, safely.

    Args:
        require_positive: True for the LB/UB thresholds (must be strictly
            positive to be valid); False for a conversion factor, which must
            just be non-zero (but could legitimately be negative).

    Returns:
        float, or None if the row/column does not exist, if the value is NaN,
        or if it fails the require_positive test.
    """
    try:
        if row_name not in thresholds_df.index or column_name not in thresholds_df.columns:
            return None
        value = thresholds_df.loc[row_name, column_name]
        if pd.isna(value):
            return None
        value = float(value)
        if require_positive and value <= 0:
            return None
        if not require_positive and value == 0:
            return None
        return value
    except (KeyError, TypeError, ValueError):
        return None


def lookup_first_text(thresholds_df, row_names, column_name):
    """Try several candidate row names in order (useful because thresholds.xlsx
    is not always spelled identically), return the first non-NaN text found,
    or "" otherwise."""
    if isinstance(row_names, str):
        row_names = [row_names]
    try:
        for row_name in row_names:
            if row_name in thresholds_df.index and column_name in thresholds_df.columns:
                val = thresholds_df.loc[row_name, column_name]
                if val is not None and str(val).lower() != "nan":
                    return str(val)
    except (KeyError, TypeError):
        pass
    return ""


def extract_value(payload, value_key):
    """Footprint of an (LP, scenario): CBA = domestic + imported; PBA =
    payload["pba"] (already a scalar). None if missing/non-finite."""
    if payload is None:
        return None
    if value_key == "pba":
        val = payload.get("pba")
    else:
        val = float(payload[DOMESTIC].sum() + payload[IMPORTED].sum())
    return float(val) if val is not None and np.isfinite(val) else None


def mean_variation(scenario_data, reference_data, value_keys):
    """Mean relative variation (e.g. -0.3 = -30 %) of the footprint of a
    scenario vs the reference, over all LPs and all requested accountings
    (value_keys, "cba" and/or "pba") -- None if no comparable value."""
    variations = []
    for indicator_name, ref_payload in reference_data.items():
        for key in value_keys:
            ref_val = extract_value(ref_payload, key)
            val = extract_value(scenario_data.get(indicator_name), key)
            if ref_val and val is not None:
                variations.append((val - ref_val) / abs(ref_val))
    return float(np.mean(variations)) if variations else None


def sort_scenarios_by_reduction(all_scenarios_data, scenario_names, reference_idx, value_keys):
    """Display order (indices): reference first, then the trend scenario (name
    = config.TREND_SCENARIO_PATTERN), then the other scenarios by decreasing
    mean variation (mean_variation): the smallest decrease / largest increase
    right next to the trend scenario, the largest decrease the furthest away."""
    reference_data = all_scenarios_data[reference_idx]
    others = [i for i in range(len(scenario_names)) if i != reference_idx]
    trend = [i for i in others
             if re.match(config.TREND_SCENARIO_PATTERN, scenario_names[i].strip(), re.IGNORECASE)]
    rest = [i for i in others if i not in trend]

    def sort_key(i):
        variation = mean_variation(all_scenarios_data[i], reference_data, value_keys)
        return -np.inf if variation is None else variation

    return [reference_idx] + trend + sorted(rest, key=sort_key, reverse=True)


def lookup_dls(dls_df, dls_row_name, indicator_name):
    """National (France) DLS social floor, in "Figures unit", read as is on the
    dls_row_name row of the DLS sheet (e.g. "DLS", "DLS 2050 upper bound" --
    the "... (p.cap)" per-capita rows are separate and ignored).

    Returns:
        float, or None if the row/column is missing or empty.
    """
    # The labels of the DLS sheet are not always clean ("DLS ")
    stripped_index = {row.strip(): row for row in dls_df.index if isinstance(row, str)}
    if dls_row_name not in stripped_index:
        return None
    return lookup_cell(dls_df, stripped_index[dls_row_name], indicator_name, require_positive=True)


def lookup_threshold(thresholds_df, threshold_row_name, indicator_name):
    """LOWER/LB/UB threshold read as is in thresholds.xlsx (rows "Equality
    (Lower bound)"/"Equality"/"Equality (Upper bound)"), strictly positive or None."""
    return lookup_cell(thresholds_df, threshold_row_name, indicator_name, require_positive=True)


# ============================================================================
# POPULATION-BASED COMPUTATIONS ("Population" sheet of budget_shares.xlsx)
# ============================================================================


def lookup_population(pop_df, period, region_code):
    """Population in PERSONS of `region_code` for a period ("2019", "2019-2050"...)
    of the "Population" sheet (which stores thousands, see io.load_population_df)."""
    return float(pop_df.loc[region_code, str(period)]) * 1000




# ============================================================================
# BUDGETS BY REGION FROM SHARES (budget_shares.xlsx)
# ============================================================================
# Descent from the world budget to the budget of a country/region: the world
# budget is multiplied by a share read in budget_shares.xlsx ("EPC" = equal
# per capita and "CTR" = capability to reduce principles, each with a
# reference share and min/max bounds over the SSP variants). The budgets
# obtained are in absolute values ("Figures unit"), never per capita.

# threshold_kind -> row of thresholds.xlsx/"All" giving the world budget.
_WORLD_BUDGET_ROW_FOR_KIND = {
    "lower": config.WORLD_BUDGET_ROW_LOWER,
    "lb": config.WORLD_BUDGET_ROW_LB,
    "ub": config.WORLD_BUDGET_ROW_UB,
}


def resolve_region_code(scenario_name, codes, default=config.DEFAULT_REGION_CODE):
    """EXIOBASE region code (among `codes`: the index of shares_df, of pop_df...) matching a scenario name.

    The folder name is split on every non-alphanumeric character and each
    token is compared with the available codes, case-insensitively:
    "2019_EU27" -> "EU27", "2019_FR" -> "FR", "2019_W" -> "W". The comparison
    is made on whole tokens and not on a substring, so as not to match "TREND"
    with "TR" (Türkiye) or "Tech_NZE" with "CH" (Switzerland).

    Scenarios whose name contains no code (Base_year, TREND, Tech_NZE,
    Sufficiency_NZE...) are France scenarios: they get `default`
    (config.DEFAULT_REGION_CODE, i.e. "FR").

    `codes` may also be a shares DataFrame (io.load_budget_shares_df), whose
    index is then used.
    """
    if isinstance(codes, pd.DataFrame):
        codes = codes.index
    codes = {str(code).upper(): code for code in codes}
    for token in re.split(r"[^A-Za-z0-9]+", str(scenario_name)):
        if token and token.upper() in codes:
            return codes[token.upper()]
    return default


def find_share_column(shares_df, sharing_principle, variant):
    """Share column matching (sharing principle, variant), or None.

    Args:
        sharing_principle: "EPC" or "CTR".
        variant: "ref" (reference share, which positions the bubbles), "min"
            or "max" (bounds of the sensitivity interval).

    The exact name of the reference columns contains details that may change
    (chosen SSP, period): they are therefore found by pattern
    (config.BUDGET_SHARE_COLUMN_PATTERNS) rather than by exact name.
    """
    if sharing_principle not in config.BUDGET_SHARE_COLUMN_PATTERNS:
        raise ValueError(
            f"Unknown sharing_principle: {sharing_principle!r} "
            f"(expected: {', '.join(config.BUDGET_SHARE_COLUMN_PATTERNS)})"
        )
    pattern = config.BUDGET_SHARE_COLUMN_PATTERNS[sharing_principle][variant]
    for column in shares_df.columns:
        if re.search(pattern, str(column), flags=re.IGNORECASE):
            return column
    return None


def lookup_budget_share(shares_df, region_code, sharing_principle, variant="ref",
                        column_pattern=None):
    """Share of the world budget going to `region_code`, as a float, or None if
    it is missing/cannot be computed (column not found, unknown region code,
    "n.a." or empty cell -- see io.load_budget_shares_df).

    column_pattern: regex pattern directly designating the column to read,
    instead of the one derived from (sharing_principle, variant)."""
    if column_pattern is not None:
        column = next((c for c in shares_df.columns
                       if re.search(column_pattern, str(c), flags=re.IGNORECASE)), None)
    else:
        column = find_share_column(shares_df, sharing_principle, variant)
    if column is None or region_code not in shares_df.index:
        return None
    value = shares_df.loc[region_code, column]
    if isinstance(value, pd.Series):  # region code duplicated in the sheet
        value = value.iloc[0]
    if pd.isna(value):
        return None
    value = float(value)
    return value if value > 0 else None


def compute_region_budget(thresholds_df, shares_df, indicator_name, region_code,
                          sharing_principle, variant="ref", threshold_kind="lb",
                          column_pattern=None):
    """LOWER/LB/UB budget of a country/region, in absolute values, obtained by
    applying to the world budget the share of `region_code` read in shares_df.

    Formula:
        region_budget = World_budget(LOWER, LB or UB) / Budget_conversion * Region_share

    where World_budget is read on the rows "Lower safe bound"/"Safe limit"/
    "Upper safe bound" of thresholds.xlsx/"All" (unit "Unit budget") and
    Budget_conversion on "Unit conversion budget" (-> "Figures unit").

    Args:
        threshold_kind: "lower", "lb" or "ub".
        variant: "ref", "min" or "max" (see find_share_column).
        column_pattern: regex pattern of the share column to read, taking
            precedence over (sharing_principle, variant) -- see lookup_budget_share.

    Returns:
        float, or None if a required piece of data is missing (world budget
        not filled in for this indicator, missing conversion, unknown share).
    """
    if threshold_kind not in _WORLD_BUDGET_ROW_FOR_KIND:
        raise ValueError(f"Unknown threshold_kind: {threshold_kind!r}")

    world_budget = lookup_cell(thresholds_df, _WORLD_BUDGET_ROW_FOR_KIND[threshold_kind],
                               indicator_name, require_positive=True)
    world_conversion = lookup_cell(thresholds_df, config.WORLD_CONVERSION_ROW_ABS, indicator_name)
    share = lookup_budget_share(shares_df, region_code, sharing_principle, variant, column_pattern)
    if world_budget is None or not world_conversion or share is None:
        return None

    return world_budget / world_conversion * share


def compute_world_period_budget(thresholds_df, pop_df, indicator_name, period,
                                threshold_kind="lb",
                                pressure_period=config.WORLD_PRESSURE_POPULATION_PERIOD):
    """"Equivalent" LOWER/LB/UB world budget (absolute values) when the limit
    is reasoned per capita over a given population period.

    Used for the sensitivity of W, whose share is 1 without min/max bounds in
    budget_shares.xlsx. The per-capita limit (Budget / World_pop(period)) is
    compared with the world per-capita pressures (Pressures / World_pop(2019)):

        overshoot_pc = (Pressures / Pop_2019) / (Budget / Pop_period)
                     = Pressures / (Budget * Pop_2019 / Pop_period)

    hence the returned equivalent budget, Budget * Pop_2019 / Pop_period. For
    period = "2019" this is exactly compute_region_budget with a share of 1.

    Returns:
        float, or None if a required piece of data is missing.
    """
    if threshold_kind not in _WORLD_BUDGET_ROW_FOR_KIND:
        raise ValueError(f"Unknown threshold_kind: {threshold_kind!r}")

    world_budget = lookup_cell(thresholds_df, _WORLD_BUDGET_ROW_FOR_KIND[threshold_kind],
                               indicator_name, require_positive=True)
    world_conversion = lookup_cell(thresholds_df, config.WORLD_CONVERSION_ROW_ABS, indicator_name)
    if world_budget is None or not world_conversion:
        return None

    pressure_population = lookup_population(pop_df, pressure_period, config.WORLD_REGION_CODE)
    period_population = lookup_population(pop_df, period, config.WORLD_REGION_CODE)
    return world_budget / world_conversion * pressure_population / period_population


# ============================================================================
# FOOTPRINT IN ABSOLUTE VALUES (multi-scenario Synthesis, multi-scenario Overshoot)
# ============================================================================


def process_single_indicator_scenario(indicator_name, lp_name, scenario_folder_path,
                                        cf_df, bridge_matrices_df, thresholds_df):
    """Footprint (dom + imp, + F_Y if applicable) of a indicator for a given
    LP and scenario, broken down by consumption category through the bridge
    matrix. Also computes "pba" (production-based, F_x_dom + F_Y, a single
    value not broken down by category). Building block of process_scenario()."""
    d_cba_imp = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="imp")
    d_cba_dom = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="dom")
    f_y_tot_df = io.load_f_y_tot_france(scenario_folder_path, lp_name)
    f_x_dom_df = io.load_f_x_dom_france(scenario_folder_path, lp_name)

    if d_cba_imp is None or d_cba_dom is None:
        return None

    extension_factor_map = build_extension_factor_map(cf_df, indicator_name)

    vector_imp = filter_and_weight_vector(d_cba_imp, extension_factor_map)
    vector_dom = filter_and_weight_vector(d_cba_dom, extension_factor_map)

    if vector_imp.empty or vector_dom.empty:
        return None

    f_y_tot_value = None
    if f_y_tot_df is not None:
        f_y_tot_weighted = filter_and_weight_vector(f_y_tot_df, extension_factor_map)
        if not f_y_tot_weighted.empty:
            f_y_tot_value = float(f_y_tot_weighted.sum())

    # PBA (production-based accounting): domestic production footprint
    # (F_x_dom) + F_Y, same factors and conversion as the CBA below, but
    # without breakdown by category or by dom/imp (a single value).
    pba_raw = filter_and_weight_scalar(f_x_dom_df, extension_factor_map) if f_x_dom_df is not None else 0.0
    if f_y_tot_value is not None:
        pba_raw += f_y_tot_value

    # The bridge breaks down by consumption category; it only applies to the
    # France scenarios (World/Europe remain a single aggregated total).
    scen_name = str(scenario_folder_path.name)
    if not (scen_name.endswith("Europe") or scen_name.endswith("World")):
        bridge_categories = bridge_matrices_df.iloc[1:, 6:]
        bridge_categories.index = bridge_matrices_df.iloc[1:, 5].values
        result_imp = bridge_categories.T @ vector_imp
        result_dom = bridge_categories.T @ vector_dom
    else:
        result_imp = vector_imp.copy()
        result_dom = vector_dom.copy()

    conversion_factor = lookup_cell(thresholds_df, config.CONVERSION_ROW_ABS, indicator_name)
    if conversion_factor is not None:
        result_imp = result_imp / conversion_factor
        result_dom = result_dom / conversion_factor
    pba_value = pba_raw / conversion_factor if conversion_factor else pba_raw

    # F_Y: direct emissions of the final demand, added as a category of its
    # own (not broken down by the bridge, already attached to final consumption).
    if f_y_tot_value is not None:
        if conversion_factor:
            f_y_tot_value = f_y_tot_value / conversion_factor
        result_dom = result_dom.copy()
        result_imp = result_imp.copy()
        result_dom["F_Y"] = f_y_tot_value
        result_imp["F_Y"] = 0.0

    return {
        DOMESTIC: result_dom,
        IMPORTED: result_imp,
        "categories": result_imp.index.tolist(),
        "pba": pba_value,
    }


def process_scenario(scenario_folder_path, cf_df, bridge_matrices_df, thresholds_df):
    """Process a complete scenario: aggregate all the LPs of a same indicator.

    Returns:
        tuple: (data_by_indicator: {indicator_name: result}, indicator_to_lp)
    """
    indicator_to_lp = get_unique_indicators(cf_df)
    data_by_indicator = {}

    for indicator_name, lp_list in indicator_to_lp.items():
        aggregated = None
        for lp_name in lp_list:
            result = process_single_indicator_scenario(
                indicator_name, lp_name, scenario_folder_path,
                cf_df, bridge_matrices_df, thresholds_df,
            )
            if result is None:
                continue
            if aggregated is None:
                aggregated = {
                    DOMESTIC: result[DOMESTIC].copy(),
                    IMPORTED: result[IMPORTED].copy(),
                    "categories": result["categories"],
                    "pba": result["pba"],
                }
            else:
                aggregated[DOMESTIC] = aggregated[DOMESTIC].add(result[DOMESTIC], fill_value=0)
                aggregated[IMPORTED] = aggregated[IMPORTED].add(result[IMPORTED], fill_value=0)
                aggregated["pba"] += result["pba"]

        if aggregated is not None:
            aggregated["categories"] = colors.sort_categories(aggregated[IMPORTED].index.tolist())
            data_by_indicator[indicator_name] = aggregated

    return data_by_indicator, indicator_to_lp


def process_indicator_lp_breakdown(indicator_name, scenario_folder_path, cf_df, thresholds_df):
    """Absolute footprint (dom + imp + F_Y) of a indicator for a scenario,
    broken down by LP (Earth-system process) of origin -- unlike
    process_single_indicator_scenario/process_scenario, which break down by
    consumption category by summing all the LPs together, this function keeps
    each LP separate and does not break down by category.

    Used by plot_process_breakdown to show the share of each LP in the
    footprint of a given indicator (e.g. Biodiversity loss).
    Same dom/imp/World-Europe loading logic as _scenario_footprint_totals.

    Returns:
        {lp_name: absolute_value} -- only the LPs for which data exist for
        this scenario (possibly empty dict, a missing LP is never explicitly
        set to 0).
    """
    indicator_to_lp = get_unique_indicators(cf_df)
    lp_list = indicator_to_lp.get(indicator_name, [])

    is_world_europe = not scenario_has_dom_imp_split(scenario_folder_path)
    conversion_factor = lookup_cell(thresholds_df, config.CONVERSION_ROW_ABS, indicator_name)

    lp_values = {}
    for lp_name in lp_list:
        if is_world_europe and lp_name == "ghg_combustion":
            continue

        ext_map = build_extension_factor_map(cf_df, indicator_name, lp_name)
        total = 0.0
        has_data = False

        if is_world_europe:
            d_cba = io.load_d_cba_world_europe(scenario_folder_path, lp_name)
            if d_cba is not None:
                total += filter_and_weight_scalar(d_cba, ext_map)
                has_data = True
            f_y_tot = io.load_f_y_tot_world_europe(scenario_folder_path, lp_name)
            if f_y_tot is not None:
                total += filter_and_weight_scalar(f_y_tot, ext_map)
                has_data = True
        else:
            for origin in ("dom", "imp"):
                d_cba = io.load_d_cba_france(scenario_folder_path, lp_name, origin)
                if d_cba is not None:
                    total += filter_and_weight_scalar(d_cba, ext_map)
                    has_data = True
            f_y_tot = io.load_f_y_tot_france(scenario_folder_path, lp_name)
            if f_y_tot is not None:
                total += filter_and_weight_scalar(f_y_tot, ext_map)
                has_data = True

        if has_data:
            lp_values[lp_name] = total / conversion_factor if conversion_factor else total

    return lp_values


# ============================================================================
# TOTAL CBA + PBA FOOTPRINT OF A SCENARIO (absolute values or per capita)
# ============================================================================


def scenario_has_dom_imp_split(scenario_folder_path):
    """True if the extensions of the scenario are split into dom_{lp}/imp_{lp}
    (France scenarios), False if they are already aggregated into a single
    folder per LP (World/Europe scenarios, see reformat_d_cba_monde_europe.py).

    Detection based on the folder structure and not on its name: scenario
    folder names change (2019_World -> 2019_W, 2019_Europe_27 -> 2019_EU27...),
    not their tree structure.
    """
    extensions_dir = Path(scenario_folder_path) / "extensions"
    if not extensions_dir.is_dir():
        return False
    return any(d.is_dir() and d.name.startswith(("dom_", "imp_")) for d in extensions_dir.iterdir())


def _scenario_footprint_totals(scenario_folder_path, cf_df, is_world_europe):
    """CBA and PBA footprint totals by indicator, in raw Exiobase units
    (not converted) -- common base of process_scenario_absolute and
    process_scenario_per_capita.

    CBA adds dom + imp (France) or reads the single file (World/Europe); PBA
    only uses the domestic part. F_Y_tot.pkl, if present for an LP, is added
    to both totals. ghg_combustion is ignored for World/Europe (missing from
    these data).

    Returns:
        {indicator: (total_cba, total_pba)} -- only the indicators for
        which at least one file exists.
    """
    indicator_to_lp = get_unique_indicators(cf_df)
    totals = {}

    for indicator_name, lp_list in indicator_to_lp.items():
        total_cba = 0.0
        total_pba = 0.0
        has_data = False

        for lp_name in lp_list:
            if is_world_europe and lp_name == "ghg_combustion":
                continue

            ext_map = build_extension_factor_map(cf_df, indicator_name, lp_name)

            if is_world_europe:
                d_cba = io.load_d_cba_world_europe(scenario_folder_path, lp_name)
                if d_cba is not None:
                    total_cba += filter_and_weight_scalar(d_cba, ext_map)
                    has_data = True
                f_dom = io.load_f_x_dom_world_europe(scenario_folder_path, lp_name)
                if f_dom is not None:
                    total_pba += filter_and_weight_scalar(f_dom, ext_map)
                f_y_tot = io.load_f_y_tot_world_europe(scenario_folder_path, lp_name)
                if f_y_tot is not None:
                    f_y_tot_total = filter_and_weight_scalar(f_y_tot, ext_map)
                    total_cba += f_y_tot_total
                    total_pba += f_y_tot_total
                    has_data = True
            else:
                for origin in ("dom", "imp"):
                    d_cba = io.load_d_cba_france(scenario_folder_path, lp_name, origin)
                    if d_cba is not None:
                        total_cba += filter_and_weight_scalar(d_cba, ext_map)
                        has_data = True
                f_dom = io.load_f_x_dom_france(scenario_folder_path, lp_name)
                if f_dom is not None:
                    total_pba += filter_and_weight_scalar(f_dom, ext_map)
                f_y_tot = io.load_f_y_tot_france(scenario_folder_path, lp_name)
                if f_y_tot is not None:
                    f_y_tot_total = filter_and_weight_scalar(f_y_tot, ext_map)
                    total_cba += f_y_tot_total
                    total_pba += f_y_tot_total
                    has_data = True

        if has_data:
            totals[indicator_name] = (total_cba, total_pba)

    return totals


def _pack_totals(cba, pba):
    """Put a (CBA, PBA) pair in the payload format expected by the overshoot
    figures (_total_footprint sums DOMESTIC + IMPORTED)."""
    return {
        DOMESTIC: pd.Series([cba], index=["total"]),
        IMPORTED: pd.Series([0.0], index=["total"]),
        "pba": pba,
        "categories": ["total"],
    }


def process_scenario_absolute(scenario_folder_path, cf_df, thresholds_df):
    """Total CBA and PBA footprint of a scenario, in absolute values ("Figures
    unit" of thresholds.xlsx), for all indicators.

    Formula: footprint = exiobase_total / Conversion ("Unit conversion Exiobase").
    Intended for plot_overshoot.create_overshoot_safe_space_figure_by_region,
    where the footprint of a country is compared with its share of the world
    budget.

    Unlike process_scenario (also in absolute values, but restricted to the
    France scenarios since it breaks down the footprint by consumption
    category through the bridge matrix), this function only returns one total
    per indicator and also accepts the World/Europe scenarios.

    Returns:
        {indicator: {DOMESTIC: Series([cba], index=["total"]),
                       IMPORTED: Series([0.0], index=["total"]),
                       "pba": pba (float), "categories": ["total"]}}
    """
    is_world_europe = not scenario_has_dom_imp_split(scenario_folder_path)
    totals = _scenario_footprint_totals(scenario_folder_path, cf_df, is_world_europe)

    data_by_indicator = {}
    for indicator_name, (total_cba, total_pba) in totals.items():
        conversion = lookup_cell(thresholds_df, config.CONVERSION_ROW_ABS, indicator_name)
        if conversion is not None:
            total_cba, total_pba = total_cba / conversion, total_pba / conversion
        data_by_indicator[indicator_name] = _pack_totals(total_cba, total_pba)

    return data_by_indicator


def get_population(scenario_path, pop_df):
    """Population in PERSONS for a scenario. Period deduced from the folder name
    ("2019" / "2015" if present, otherwise "2050", see
    config.SCENARIO_POPULATION_PERIODS), region from the region code contained
    in the name (whole tokens, see resolve_region_code -- "2019_W" -> W,
    "2019_EU27" -> EU27, "2019_WF" -> WF, otherwise France)."""
    path_str = str(scenario_path)
    period = next((p for p in config.SCENARIO_POPULATION_PERIODS if p in path_str),
                  config.DEFAULT_SCENARIO_POPULATION_PERIOD)
    region_code = resolve_region_code(Path(scenario_path).name, pop_df.index)
    return lookup_population(pop_df, period, region_code)


def process_scenario_per_capita(scenario_folder_path, cf_df, thresholds_df, pop_df):
    """Per-capita CBA (consumption-based, d_cba) and PBA (production-based,
    F_x_dom) footprint for all the indicators of a scenario.

    Formula: footprint_per_cap = exiobase_total / Conversion(p.cap) / population.
    Same totals as process_scenario_absolute (_scenario_footprint_totals), with
    the "(p.cap)" conversion row and the population of get_population.

    Returns:
        {indicator: {DOMESTIC: Series([cba_per_cap], index=["total"]),
                       IMPORTED: Series([0.0], index=["total"]),
                       "pba": pba_per_cap (float), "categories": ["total"]}}
    """
    is_world_europe = not scenario_has_dom_imp_split(scenario_folder_path)
    population = get_population(scenario_folder_path, pop_df)
    totals = _scenario_footprint_totals(scenario_folder_path, cf_df, is_world_europe)

    data_by_indicator = {}
    for indicator_name, (total_cba, total_pba) in totals.items():
        conversion = lookup_cell(thresholds_df, config.CONVERSION_ROW_PER_CAPITA, indicator_name)
        divisor = population * (conversion if conversion is not None else 1.0)
        data_by_indicator[indicator_name] = _pack_totals(total_cba / divisor, total_pba / divisor)

    return data_by_indicator
