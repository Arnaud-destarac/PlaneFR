"""
Loading of the source data: Excel files (characterization factors, bridge
matrices, thresholds, budget shares, population) and MRIO pickles by scenario
(d_cba, F_x_dom, F_Y_tot, M).

General convention of the pickle loaders: they return None if the file does
not exist for this scenario/LP (rather than raising an exception), because a
missing file for some scenario x indicator combinations is a normal case
(e.g. ghg_combustion missing from the World/Europe scenarios).
"""

import re
from pathlib import Path

import pandas as pd

from . import config

# ============================================================================
# EXCEL FILES (common to all scenarios)
# ============================================================================


AGRI_GROUP_SUFFIX = "_agri"


def is_agri_group(scenario_group):
    """True if the scenario group is an agricultural group ("{scenario}_agri")."""
    return isinstance(scenario_group, str) and scenario_group.endswith(AGRI_GROUP_SUFFIX)


def _normalize_label(label):
    """Collapse repeated whitespace and strip the ends of a row/column label
    (non-string labels are returned unchanged)."""
    return re.sub(r"\s+", " ", label).strip() if isinstance(label, str) else label


def load_characterization_factors(scenario_group=None):
    """Characterization factors: one row per (indicator, LP, EXIOBASE stressor),
    with columns config.CF_INDICATOR_COL ("Indicators"), config.CF_LP_COL
    ("EXIOBASE groups"), config.CF_STRESSOR_COL ("EXIOBASE stressors"),
    config.CF_UNIT_COL and config.CF_FACTOR_COL ("Characterization factors").

    Reads the config.CF_SHEET sheet ("All"), or config.CF_AGRI_SHEET ("Agri") if
    scenario_group ends with "_agri". Column names are matched to the config
    constants case-insensitively (the two sheets do not use the same
    capitalization, e.g. "EXIOBASE Groups" / "EXIOBASE groups"), and rows
    without a indicator or stressor are dropped."""
    sheet = config.CF_AGRI_SHEET if is_agri_group(scenario_group) else config.CF_SHEET
    df = pd.read_excel(config.CHARACTERIZATION_FACTORS_FILE, sheet_name=sheet)
    canonical = {name.casefold(): name for name in (
        config.CF_INDICATOR_COL, config.CF_LP_COL, config.CF_STRESSOR_COL,
        config.CF_UNIT_COL, config.CF_FACTOR_COL)}
    df = df.rename(columns=lambda c: canonical.get(_normalize_label(str(c)).casefold(), c))
    missing = [c for c in canonical.values() if c not in df.columns and c != config.CF_UNIT_COL]
    if missing:
        raise KeyError(f"Columns {missing} not found in sheet '{sheet}' of "
                       f"{config.CHARACTERIZATION_FACTORS_FILE.name} (columns: {list(df.columns)})")
    df = df.dropna(subset=[config.CF_INDICATOR_COL, config.CF_STRESSOR_COL])
    for col in (config.CF_INDICATOR_COL, config.CF_LP_COL, config.CF_STRESSOR_COL):
        df[col] = df[col].map(_normalize_label)
    return df


def load_bridge_matrices():
    """'Final conso. cat.' bridge matrix: sector -> consumption category mapping
    (Food/Housing/Mobility/Final goods/Final services), for the stacked bars and
    the overshoot figure in absolute values."""
    return pd.read_excel(config.BRIDGE_MATRICES_FILE, sheet_name="Final conso. cat.")


def load_bridge_m_matrix(sheet_name="M"):
    """Products -> consumption categories bridge (200 products x k categories)
    for the aggregation by category of the Excel tables / of the bubble chart.

    Args:
        sheet_name: "M" (5 categories: Food, Housing, Mobility, Final goods,
            Final services) or "M_split" (finer breakdown, 12 categories).
            All the columns after "sector" are read as categories.

    Not to be confused with the 'Alimentation' sheet of the same workbook,
    which is a fine breakdown of food products (rice, beef, pork...).

    The 1st data row of the sheet is a 2nd header row (category /
    sub_category / sector): the 200 products are on rows 1 to 200."""
    m_raw = pd.read_excel(config.BRIDGE_MATRICES_FILE, sheet_name=sheet_name)
    # Category columns: those after "sector", excluding empty columns without header.
    category_cols = [c for c in m_raw.columns[3:] if not str(c).startswith("Unnamed")]
    bridge_m = m_raw.loc[m_raw.index[1:201], category_cols].copy()
    bridge_m.index = m_raw.iloc[1:201, 2].values
    bridge_m = bridge_m.apply(pd.to_numeric, errors="coerce").fillna(0)
    return (bridge_m > 0).astype(int)


def load_thresholds(scenario_group=None):
    """config.THRESHOLDS_SHEET sheet ("All") of thresholds.xlsx: planetary
    thresholds/limits by indicator (one column per indicator, one row per
    notion). If scenario_group ends with "_agri", reads config.THRESHOLDS_AGRI_SHEET
    ("Agri") instead.

    Row names are normalized (repeated whitespace collapsed to a single space,
    ends stripped): the sheet contains several labels with double spaces or
    trailing spaces (e.g. "Unit conversion Exiobase  (p.cap)"), which would
    otherwise break the lookups by exact name (see planefr_lib.config)."""
    sheet = config.THRESHOLDS_AGRI_SHEET if is_agri_group(scenario_group) else config.THRESHOLDS_SHEET
    df = pd.read_excel(config.THRESHOLDS_FILE, sheet_name=sheet, index_col=0)
    df.index = df.index.map(_normalize_label)
    df.columns = df.columns.map(_normalize_label)
    return df


def load_dls():
    """config.DLS_SHEET sheet of thresholds.xlsx: social floors (decent living standards)."""
    return pd.read_excel(config.THRESHOLDS_FILE, sheet_name=config.DLS_SHEET, index_col=0)


def load_population_df():
    """config.POPULATION_SHEET sheet of budget_shares.xlsx: UN population (WPP
    2024) in thousands, indexed by EXIOBASE region code (FR, EU27, W...), one
    column per period, renamed to the period itself ("2015", "2019", "2050",
    "2019-2050", "2019-2100": mean over the period)."""
    df = pd.read_excel(config.BUDGET_SHARES_FILE, sheet_name=config.POPULATION_SHEET,
                       header=config.BUDGET_SHARES_HEADER_ROW)
    df = df[df[config.BUDGET_SHARE_CODE_COLUMN].notna() & df[config.BUDGET_SHARE_REGION_COLUMN].notna()]
    df = df.set_index(config.BUDGET_SHARE_CODE_COLUMN)
    df.index = df.index.map(lambda x: str(x).strip())
    periods = {}
    for col in df.columns:
        match = re.search(config.POPULATION_COLUMN_PATTERN, str(col), flags=re.IGNORECASE)
        if match:
            periods[col] = match.group(1)
    return df[list(periods)].rename(columns=periods).apply(pd.to_numeric, errors="coerce")


def bdf_file(scenario_group=None):
    """File impacts_facteurs_agrégé_{scenario_group}*.xlsx produced by BdF.ps1
    (e.g. group "Base_year" -> impacts_facteurs_agrégé_Base_year_2019_FR.xlsx).
    scenario_group None = config.SCENARIO_GROUP. Exactly one file must match."""
    if scenario_group is None:
        scenario_group = config.SCENARIO_GROUP
    base = config.BDF_FILE
    exact = base.with_name(f"{base.stem}_{scenario_group}{base.suffix}")
    if exact.is_file():
        return exact
    matches = sorted(base.parent.glob(f"{base.stem}_{scenario_group}_*{base.suffix}"))
    if len(matches) != 1:
        found = [p.name for p in matches] or "none"
        raise FileNotFoundError(
            f"BdF file of group '{scenario_group}' not found or ambiguous in {base.parent} "
            f"(expected: {base.stem}_{scenario_group}[_...]{base.suffix}; found: {found})")
    return matches[0]


def _is_base_year_scenario(scenario_name):
    """"Base year" scenario folder (2019_FR, Base_year_2015...)."""
    return "2019" in scenario_name or scenario_name.startswith("Base_year")


def bdf_comparison_file(scenario_group=None):
    """BdF file of the comparison scenario of a 2-scenario group (e.g. group
    "S1+" = 2019_FR + S1_2050): the "base year" scenario of the group, read in
    impacts_facteurs_agrégé_{group}_{scenario}.xlsx if it exists, otherwise in
    the file of the config.DLS_BASE_YEAR_GROUP group (impacts_facteurs_agrégé_Base_year*.xlsx).
    The file of the group itself (bdf_file) holds the other scenario (e.g. S1_2050).

    Returns:
        Path, or None if the group contains only one scenario (or is itself
        the base-year group).
    """
    if scenario_group is None:
        scenario_group = config.SCENARIO_GROUP
    group_dir = config.BASE_DATA_DIR / scenario_group
    if not scenario_group or scenario_group == config.DLS_BASE_YEAR_GROUP or not group_dir.is_dir():
        return None
    scenarios = sorted(d.name for d in group_dir.iterdir() if config._is_scenario_folder(d))
    if len(scenarios) < 2:
        return None
    base = [s for s in scenarios if _is_base_year_scenario(s)]
    if len(scenarios) != 2 or len(base) != 1:
        raise ValueError(
            f"BdF comparison: group '{scenario_group}' must contain exactly 2 scenarios "
            f"including 1 base year (found: {scenarios})")
    stem, suffix = config.BDF_FILE.stem, config.BDF_FILE.suffix
    specific = config.BDF_FILE.with_name(f"{stem}_{scenario_group}_{base[0]}{suffix}")
    return specific if specific.is_file() else bdf_file(config.DLS_BASE_YEAR_GROUP)


def load_bdf(sheet_name="décile", scenario_group=None, path=None):
    """Sheet sheet_name of impacts_facteurs_agrégé_{scenario_group}*.xlsx (see
    bdf_file): 1 row per household category (+ a config.BDF_TOTAL_ROW row),
    indexed by the 1st column; column config.BDF_WEIGHT_COL (share of the
    population) then 1 column per indicator, named "<indicator> (<unit>)".
    The sheet names ("décile", "unité urbaine", "croisé") are those written by
    the BdF pipeline (Github/ademe_eemrio). path: file to read instead of
    bdf_file(scenario_group) (e.g. bdf_comparison_file)."""
    df = pd.read_excel(path or bdf_file(scenario_group), sheet_name=sheet_name, index_col=0)
    df.index = df.index.map(str)
    return df


def load_budget_shares_df():
    """config.BUDGET_SHARES_SHEET sheet of budget_shares.xlsx: share of the world
    budget going to each EXIOBASE region (+ the EU27 aggregate), indexed by
    region code (config.BUDGET_SHARE_CODE_COLUMN: FR, EU27, W, CN...).

    The title/note rows around the table are discarded (headers read at
    config.BUDGET_SHARES_HEADER_ROW, then only the rows with both a code and a
    region name are kept -- the notes below the table occupy the code column
    but not the region one), and the shares are converted to numbers: the sheet
    writes "n.a." for the shares that cannot be computed (Taiwan, missing from
    the World Bank data) and leaves empty the min/max that make no sense ("W"
    row), which therefore become NaN."""
    def read(sheet):
        d = pd.read_excel(config.BUDGET_SHARES_FILE, sheet_name=sheet,
                          header=config.BUDGET_SHARES_HEADER_ROW)
        d = d[d[config.BUDGET_SHARE_CODE_COLUMN].notna() & d[config.BUDGET_SHARE_REGION_COLUMN].notna()]
        d = d.set_index(config.BUDGET_SHARE_CODE_COLUMN)
        d.index = d.index.map(lambda x: str(x).strip())
        return d

    df = read(config.BUDGET_SHARES_SHEET)

    # "UN – EPC <period>" columns of the detailed sheet. This sheet has a World
    # row (sum of the regions): the share of W is set to exactly 1.
    xls = pd.ExcelFile(config.BUDGET_SHARES_FILE)
    if config.BUDGET_SHARES_VARIANTS_SHEET in xls.sheet_names:
        variants = read(config.BUDGET_SHARES_VARIANTS_SHEET)
        un_columns = [c for c in variants.columns
                      if re.search(config.UN_SHARE_COLUMN_PATTERN, str(c), flags=re.IGNORECASE)]
        df = df.join(variants[un_columns])
        if config.WORLD_REGION_CODE in df.index:
            df.loc[config.WORLD_REGION_CODE, un_columns] = 1.0

    share_columns = [c for c in df.columns if c != config.BUDGET_SHARE_REGION_COLUMN]
    df[share_columns] = df[share_columns].apply(pd.to_numeric, errors="coerce")
    return df


# ============================================================================
# Y_CATEGORY
# ============================================================================


def exclude_y_category(df, excluded=("Exports",)):
    """Drop the columns whose Y_category is in `excluded` (by default the
    exports, which are not part of the French final demand).

    Only applies to the data already reformatted in data/3.x (label
    "Exports"). The reformatting scripts (reformat_d_cba_monde_europe.py) work
    on the raw upstream data, where the same type of column is labeled
    "Exports: Total (fob)" -- this is not an inconsistent filter, these are two
    different stages of the pipeline with different labels.
    """
    if "Y_category" not in df.columns.names:
        return df
    mask = ~df.columns.get_level_values("Y_category").isin(excluded)
    return df.loc[:, mask]


# ============================================================================
# D_CBA / F_X_DOM / F_Y_TOT -- FRANCE SCENARIOS (with dom/imp split)
# ============================================================================


def load_d_cba_k_france(scenario_folder_path, lp_name, origin="imp"):
    """Load {origin}_{lp}/d_cba_k.pkl (intensities per unit) for a France scenario."""
    folder_name = f"{'imp' if origin == 'imp' else 'dom'}_{lp_name}"
    file_path = Path(scenario_folder_path) / "extensions" / folder_name / "d_cba_k.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_d_cba_france(scenario_folder_path, lp_name, origin):
    """Load {origin}_{lp}/d_cba.pkl (totals, not intensities) for a France
    scenario, exports excluded."""
    file_path = Path(scenario_folder_path) / "extensions" / f"{origin}_{lp_name}" / "d_cba.pkl"
    if not file_path.exists():
        return None
    return exclude_y_category(pd.read_pickle(file_path))


def load_f_x_dom_france(scenario_folder_path, lp_name):
    """Load dom_{lp}/F_x_dom.pkl (production footprint, dom only) for a France scenario."""
    file_path = Path(scenario_folder_path) / "extensions" / f"dom_{lp_name}" / "F_x_dom.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_y_tot_france(scenario_folder_path, lp_name):
    """Load dom_{lp}/F_Y_tot.pkl (direct emissions of the final demand) for a
    France scenario, if this file exists for this LP (e.g. present for water/land)."""
    file_path = Path(scenario_folder_path) / "extensions" / f"dom_{lp_name}" / "F_Y_tot.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


# ============================================================================
# D_CBA / F_X_DOM / F_Y_TOT -- WORLD / EUROPE SCENARIOS (no dom/imp split)
# ============================================================================
# These scenarios are already aggregated into a single file per LP (see
# reformat_d_cba_monde_europe.py): their Y_category column is already reduced
# to "all", hence no need for exclude_y_category here.


def load_d_cba_world_europe(scenario_folder_path, lp_name):
    """Load {lp}/d_cba.pkl for a World/Europe scenario."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "d_cba.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_x_dom_world_europe(scenario_folder_path, lp_name):
    """Load {lp}/F_x_dom.pkl for a World/Europe scenario."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "F_x_dom.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


def load_f_y_tot_world_europe(scenario_folder_path, lp_name):
    """Load {lp}/F_Y_tot.pkl for a World/Europe scenario, if present."""
    file_path = Path(scenario_folder_path) / "extensions" / lp_name / "F_Y_tot.pkl"
    return pd.read_pickle(file_path) if file_path.exists() else None


# ============================================================================
# INTENSITY MATRICES M (bubble chart) AND FINAL DEMAND Y
# ============================================================================


def load_m_matrix(scenario_folder_path, lp_name, kind):
    """Load an environmental intensity matrix M for a given LP and type.

    Args:
        kind: "dom" (M_k.pkl of dom_{lp}), "imp" (M_k.pkl of imp_{lp}),
            "row" (M_RoW.pkl of imp_{lp}, "rest of the world" part of the imports).

    Returns:
        pd.DataFrame or None if the file does not exist.
    """
    if kind == "dom":
        folder, filename = f"dom_{lp_name}", "M_k.pkl"
    elif kind == "imp":
        folder, filename = f"imp_{lp_name}", "M_k.pkl"
    elif kind == "row":
        folder, filename = f"imp_{lp_name}", "M_RoW.pkl"
    else:
        raise ValueError("kind must be 'dom', 'imp', or 'row'")

    file_path = Path(scenario_folder_path) / "extensions" / folder / filename
    return pd.read_pickle(file_path) if file_path.exists() else None


def get_y_blocks(scenario_folder_path, excluded_y_categories=None):
    """Load system/Y.pkl for a scenario and return (y_dom_5, y_imp_5): two
    200 x 5 DataFrames (5 identical columns, duplicated for compatibility with
    the category weighting pipeline) used to size/weight the bubble chart.

    Args:
        excluded_y_categories: None (default, historical behavior of the
            bubble chart) -> sum of the 6 columns of Y.pkl; otherwise, sum of
            the columns whose Y_category (2nd level) is not in this list --
            e.g. ("GFCF", "Exports") for the final demand consistent with M_k.
    """
    y_path = Path(scenario_folder_path) / "system" / "Y.pkl"
    if not y_path.exists():
        raise FileNotFoundError(f"Y.pkl not found: {y_path}")

    y_df = pd.read_pickle(y_path)
    if excluded_y_categories is None:
        y_sum_0_6 = y_df.iloc[:, 0:6].sum(axis=1)
    else:
        keep = ~y_df.columns.get_level_values(1).isin(excluded_y_categories)
        y_sum_0_6 = y_df.loc[:, keep].sum(axis=1)

    y_5cols = pd.concat([y_sum_0_6] * 5, axis=1)
    y_5cols.columns = [f"Y_sum_0_6_{i + 1}" for i in range(5)]

    y_dom_5 = y_5cols.iloc[0:200].copy()
    y_imp_5 = y_5cols.iloc[200:400].copy()
    return y_dom_5, y_imp_5
