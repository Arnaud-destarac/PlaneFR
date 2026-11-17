"""
Paths, constants and parameters shared by all PlaneFR analyses.

thresholds.xlsx (sheet "All") contains two sets of rows for the same notions
(display unit, conversion...): one in absolute values (_ABS constants, used
by the figures) and one per capita (suffix "(p.cap)", _PER_CAPITA constants,
used by processing.process_scenario_per_capita).
"""

import os
import warnings
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_DIR / "data"
# Exiobase version of the data: can be redirected with the environment variable
# PLANEFR_EXIOBASE_VERSION (set by the matmat-ademe pipeline scripts); 3.11.2
# by default.
BASE_DATA_DIR = DATA_DIR / os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")

# Can be redirected with the environment variable PLANEFR_FIGURES_DIR (useful to
# check that a notebook runs correctly without overwriting the current
# figures); by default, the figures/ folder of the project.
FIGURES_DIR = Path(os.environ.get("PLANEFR_FIGURES_DIR", str(PROJECT_DIR / "figures")))

CHARACTERIZATION_FACTORS_FILE = DATA_DIR / "characterization_factors.xlsx"
BRIDGE_MATRICES_FILE = DATA_DIR / "bridge_matrices.xlsx"
THRESHOLDS_FILE = DATA_DIR / "thresholds.xlsx"
BUDGET_SHARES_FILE = DATA_DIR / "budget_shares.xlsx"
# Pressures (CBA footprint) broken down by household category (Budget de
# famille survey): 1 sheet per breakdown ("décile", "unité urbaine", "croisé").
# Base name: the file read is impacts_facteurs_agrégé_{group}*.xlsx (io.bdf_file).
BDF_FILE = PROJECT_DIR.parent / "Github" / "ademe_eemrio" / "impacts_facteurs_agrégé.xlsx"
BDF_WEIGHT_COL = "weight"
BDF_TOTAL_ROW = "Average"

# ============================================================================
# CHARACTERIZATION FACTORS (characterization_factors.xlsx)
# ============================================================================
# One sheet for the full set of indicators, one for the agricultural scenario
# groups ("{scenario}_agri", see io.is_agri_group).
CF_SHEET = "All"
CF_AGRI_SHEET = "Agri"

# Columns (matched case-insensitively by io.load_characterization_factors,
# since the "All" and "Agri" sheets do not use the same capitalization).
CF_INDICATOR_COL = "Indicators"                 # indicator (= thresholds.xlsx column)
CF_LP_COL = "EXIOBASE groups"                    # LP = extension folder (dom_{lp}/imp_{lp})
CF_STRESSOR_COL = "EXIOBASE stressors"           # EXIOBASE stressor (row of the MRIO tables)
CF_UNIT_COL = "Units"
CF_FACTOR_COL = "Characterization factors"

# ============================================================================
# THRESHOLDS (thresholds.xlsx)
# ============================================================================
THRESHOLDS_SHEET = "All"
THRESHOLDS_AGRI_SHEET = "Agri"
DLS_SHEET = "DLS"

# ============================================================================
# SCENARIOS
# ============================================================================

# Index of the reference scenario in get_scenario_folders(): "Base_year_2015"
# is always the first one in alphabetical order.
REFERENCE_SCENARIO_IDX = 0

# Folder name of the reference scenario (baseline), expected in every group of
# scenarios (see get_scenario_folders).
REFERENCE_SCENARIO_NAME = "Base_year_2015"

# Scenario group displayed by default: sub-folder of BASE_DATA_DIR containing
# the 2050 transition scenarios computed on a given set of accounts ("no_SSP"
# for the Base_year_2015 accounts, "SSP1_2050", "SSP2_2050"... for the accounts
# projected along an SSP, see run_engine_batch.ps1 of matmat-ademe). Can be
# redirected with the environment variable PLANEFR_SCENARIO_GROUP, or per
# notebook with the scenario_group parameter of get_scenario_folders.
# "" = old scenarios placed directly in BASE_DATA_DIR.
SCENARIO_GROUP = os.environ.get("PLANEFR_SCENARIO_GROUP", "no_SSP")


def _is_scenario_folder(path):
    """A scenario folder contains an "extensions" sub-folder (engine output;
    "system" is missing from the non-France 2019_* folders), unlike the group
    folders (no_SSP, SSP1_2050...)."""
    return path.is_dir() and (path / "extensions").is_dir()


def get_scenario_groups():
    """List the available scenario groups (sub-folders of BASE_DATA_DIR that
    are not scenario folders themselves), sorted by name."""
    return sorted(
        d.name for d in BASE_DATA_DIR.iterdir() if d.is_dir() and not _is_scenario_folder(d)
    )


def get_scenario_folders(exclude_2019=False, exclude_others=False, scenario_group=None):
    """List the scenario folders of the requested group, sorted by name.

    The displayed scenarios are all those of the folder BASE_DATA_DIR /
    scenario_group (including the reference scenario, REFERENCE_SCENARIO_NAME,
    which must therefore be there). The "2019_*" folders are always read in
    BASE_DATA_DIR.

    Args:
        exclude_2019: if True, excludes the "2019_*" folders (France/World/
            Europe_27 as alternative geographies) to keep only base year + 2050
            scenarios -- the intended behavior for the multi-scenario figures
            in absolute values. If False, includes them -- needed for the
            per-capita CBA/PBA figure, which also compares France vs Europe vs
            World.
        exclude_others: if True, keeps only the "2019_*" folders.
        scenario_group: scenario group to display ("no_SSP", "SSP1_2050"...,
            see get_scenario_groups()). None = SCENARIO_GROUP; "" = scenarios
            placed directly in BASE_DATA_DIR.

    Returns:
        list[Path]: scenario folders sorted by name.
    """
    if scenario_group is None:
        scenario_group = SCENARIO_GROUP

    root_folders = [d for d in BASE_DATA_DIR.iterdir() if _is_scenario_folder(d)]
    folders_2019 = [d for d in root_folders if "2019" in d.name]

    if scenario_group:
        group_dir = BASE_DATA_DIR / scenario_group
        if not group_dir.is_dir():
            raise ValueError(
                f"Scenario group '{scenario_group}' not found in {BASE_DATA_DIR}. "
                f"Available groups: {get_scenario_groups()}"
            )
        # All the scenarios of the group folder, and only them
        scenario_folders = [d for d in group_dir.iterdir() if _is_scenario_folder(d)]
        if REFERENCE_SCENARIO_NAME not in {d.name for d in scenario_folders}:
            warnings.warn(
                f"Group '{scenario_group}' does not contain {REFERENCE_SCENARIO_NAME}: "
                f"the reference scenario (REFERENCE_SCENARIO_IDX) will be the first one "
                f"in alphabetical order."
            )
    else:
        scenario_folders = [d for d in root_folders if "2019" not in d.name]

    if exclude_others:
        folders = folders_2019
    elif exclude_2019:
        folders = scenario_folders
    else:
        folders = scenario_folders + folders_2019
    return sorted(folders, key=lambda d: d.name)


# ============================================================================
# THRESHOLDS: row names of thresholds.xlsx (sheet "All")
# ============================================================================
# 3 thresholds per indicator, from the strictest to the loosest -- figure
# names taken from the "raw" rows of thresholds.xlsx (WORLD_BUDGET_ROW_LOWER/
# LB/UB below): LOWER ("Lower Safe Bound", row "Equality (Lower bound)") < LB
# ("Safe Limit", historical threshold, row "Equality") < UB ("Upper Safe
# Bound", row "Equality (Upper bound)").

THRESHOLD_LOWER_ABS = "Equality (Lower bound)"
THRESHOLD_LB_ABS = "Equality"
THRESHOLD_UB_ABS = "Equality (Upper bound)"
UNIT_ROW_ABS = "Figures unit"
CONVERSION_ROW_ABS = "Unit conversion Exiobase"

# Per capita (processing.process_scenario_per_capita)
UNIT_ROW_PER_CAPITA = "Figures unit (p.cap)"
CONVERSION_ROW_PER_CAPITA = "Unit conversion Exiobase (p.cap)"

# Rows of thresholds.xlsx (sheet "All") giving the LOWER/LB/UB world budget,
# in the "Unit budget" unit (world scale, e.g. GtCO2eq) -- different from the
# "Figures unit" used everywhere else in the pipeline.
WORLD_BUDGET_ROW_LOWER = "Lower safe bound"
WORLD_BUDGET_ROW_LB = "Safe limit"
WORLD_BUDGET_ROW_UB = "Upper safe bound"

# Conversion factor "Unit budget" (world) -> "Figures unit", by indicator
# (e.g. GHG emissions: GtCO2eq -> MtCO2eq): row used as a divisor, like
# CONVERSION_ROW_ABS.
WORLD_CONVERSION_ROW_ABS = "Unit conversion budget"

# ============================================================================
# POPULATION (sheet "Population" of budget_shares.xlsx, see io.load_population_df)
# ============================================================================
# UN population (thousands) by EXIOBASE region code (rows) and by period
# (columns "UN – Population <period> (thousands)", read under the key <period>).
POPULATION_SHEET = "Population"
POPULATION_COLUMN_PATTERN = r"^UN\W+Population\s+(\S+)"

# Population periods of the per-capita sensitivity of W in
# plot_overshoot.create_overshoot_safe_space_figure_by_region (the world limit
# is divided by the World population of each period, see
# processing.compute_world_period_budget).
WORLD_SENSITIVITY_POPULATION_PERIODS = ("2019", "2019-2050", "2019-2100")

# Population period of the world pressures (2019_W folder): the per-capita
# limit of a given period is compared to them by scaling it to this population
# (processing.compute_world_period_budget).
WORLD_PRESSURE_POPULATION_PERIOD = "2019"

# Population period of a scenario, by year found in its folder name
# (processing.get_population): "2019" / "2015", otherwise "2050".
SCENARIO_POPULATION_PERIODS = ("2019", "2015")
DEFAULT_SCENARIO_POPULATION_PERIOD = "2050"

# ============================================================================
# SHARES OF THE WORLD BUDGET BY EXIOBASE REGION (budget_shares.xlsx)
# ============================================================================
# The budget of a country/region is the world budget multiplied by a share
# read in budget_shares.xlsx. Used by plot_overshoot.create_overshoot_safe_space_figure_by_region
# (figure in absolute values, hence _ABS conversions: WORLD_CONVERSION_ROW_ABS /
# CONVERSION_ROW_ABS).

BUDGET_SHARES_SHEET = "Budget_shares_regions"

# Every sheet of budget_shares.xlsx starts with a title and two note rows: the
# actual header row ("EXIOBASE code", "Region", "EPC share ref. ...") is the 4th.
BUDGET_SHARES_HEADER_ROW = 3

# Column used as index (EXIOBASE region code: FR, EU27, W, CN...) and region name.
BUDGET_SHARE_CODE_COLUMN = "EXIOBASE code"
BUDGET_SHARE_REGION_COLUMN = "Region"

# Sharing principle -> patterns (regex, case-insensitive) identifying the share
# column to use for the reference (position of the bubbles) and for the two
# sensitivity bounds.
# The "ref" patterns are anchored on "... share ref" on purpose: the sheet also
# contains "CTR / EPC ratio ref." (which contains "EPC" and "ref" without being
# a share), which must not be selected.
BUDGET_SHARE_COLUMN_PATTERNS = {
    "EPC": {"ref": r"^EPC\s+share\s+ref", "min": r"^EPC\s+min$", "max": r"^EPC\s+max$"},
    "CTR": {"ref": r"^CTR\s+share\s+ref", "min": r"^CTR\s+min$", "max": r"^CTR\s+max$"},
}

# Detailed sheet of budget_shares.xlsx, whose "UN – EPC <period>" columns (EPC
# shares computed on the UN population, WPP 2024) io.load_budget_shares_df adds
# to those of BUDGET_SHARES_SHEET. The sheet name is referenced by cell in the
# formulas of thresholds.xlsx: do not rename it.
BUDGET_SHARES_VARIANTS_SHEET = "Budget_shares_variants"
UN_SHARE_COLUMN_PATTERN = r"^UN\W+EPC\b"

# Reference EPC of plot_overshoot.create_overshoot_safe_space_figure_by_region:
# UN column (regex pattern) used by default, and exceptions by indicator
# (column name of thresholds.xlsx/"All", compared case-insensitively).
# Replaces the "EPC share ref. ..." column for the reference only: the
# sensitivity bounds remain "EPC min" / "EPC max".
REGION_FIGURE_EPC_REF_PATTERN = r"^UN\W+EPC\s+2019$"
REGION_FIGURE_EPC_REF_PATTERN_BY_INDICATOR = {
    "GHG emissions": r"^UN\W+EPC\s+2019-2100$",
    "GHG emissions (2019)": r"^UN\W+EPC\s+2019-2100$",
}

# Column of thresholds.xlsx/"All" giving the world budgets (Lower / Safe limit /
# Upper) of a indicator in that same figure, when it differs from the name
# of the indicator. The footprints and the displayed unit remain those of
# the original indicator.
REGION_FIGURE_THRESHOLD_COLUMN_BY_INDICATOR = {
    "GHG emissions": "GHG emissions (2019)",
}


def _lookup_ci(mapping, indicator_name, default):
    key = str(indicator_name).strip().casefold()
    for name, value in mapping.items():
        if name.casefold() == key:
            return value
    return default


def region_figure_epc_ref_pattern(indicator_name):
    """Pattern of the reference EPC column for `indicator_name` (see above)."""
    return _lookup_ci(REGION_FIGURE_EPC_REF_PATTERN_BY_INDICATOR, indicator_name,
                      REGION_FIGURE_EPC_REF_PATTERN)


# sharing_principle="EPC" of plot_radial_synthesis.create_radial_synthesis_figure:
# France budget = world budget x share of the RADIAL_EPC_REGION_CODE row of
# budget_shares.xlsx, read in the UN column below (regex pattern), with
# exceptions by indicator (compared case-insensitively).
RADIAL_EPC_REGION_CODE = "FR"
RADIAL_EPC_PATTERN = r"^UN\W+EPC\s+2050$"
RADIAL_EPC_PATTERN_BY_INDICATOR = {
    "GHG emissions": r"^UN\W+EPC\s+2019-2100$",
    "GHG emissions (2019)": r"^UN\W+EPC\s+2019-2100$",
}


def radial_epc_pattern(indicator_name):
    """Pattern of the EPC column used by the radial figure for `indicator_name`."""
    return _lookup_ci(RADIAL_EPC_PATTERN_BY_INDICATOR, indicator_name, RADIAL_EPC_PATTERN)


def region_figure_threshold_column(indicator_name):
    """Column of thresholds.xlsx holding the budgets of `indicator_name` in
    create_overshoot_safe_space_figure_by_region (see above)."""
    return _lookup_ci(REGION_FIGURE_THRESHOLD_COLUMN_BY_INDICATOR, indicator_name, indicator_name)


# World region code in budget_shares.xlsx (share 1, without min/max bounds: its
# sensitivity is computed per capita, see processing.compute_world_period_budget).
WORLD_REGION_CODE = "W"

# Region code used for a scenario whose folder name contains no EXIOBASE code
# (Base_year, TREND, Tech_NZE, Sufficiency_NZE...: all France scenarios).
DEFAULT_REGION_CODE = "FR"

# ============================================================================
# COLORS OF THE CONSUMPTION CATEGORIES
# ============================================================================

CATEGORY_COLORS = {
    0: "#6ac1ff",  # Blue
    1: "#fba65b",  # Orange
    2: "#4fff4f",  # Green
    3: "#ff6d6d",  # Red
    4: "#d0a3fa",  # Purple
}

# Canonical display order of the categories (F_Y always last, easy to spot)
CATEGORY_ORDER = ["Food", "Housing", "Mobility", "Final goods", "Final services", "F_Y"]

# Stable color by category name (independent of the position after multi-LP aggregation)
CATEGORY_COLOR_MAP = {
    0: "#84c9fa",  # Blue
    1: "#f9c18f",  # Orange
    2: "#8cfc8c",  # Green
    3: "#fd9393",  # Red
    4: "#d0a3fa",  # Purple
    "F_Y": "#7f7f7f",
}

# ============================================================================
# COLOR OF THE 3RD THRESHOLD ("Lower Safe Bound", THRESHOLD_LOWER_ABS)
# ============================================================================
# Light green, distinct from the green of the historical threshold (LB, "Safe
# Limit") used by plot_overshoot/plot_radial_synthesis/plot_stacked_bar --
# shared here to stay consistent across the 3 figures.
LOWER_SAFE_BOUND_COLOR = "#90ee90"

# Social floor (DLS): shared by plot_radial_synthesis/plot_stacked_bar.
# Shades of blue, from the lightest (DLS lower bound) to the darkest (DLS
# upper bound), around the historical sky blue of the central DLS.
DLS_COLOR = "#00d5ff"
DLS_LOWER_COLOR = "#20a4be"
DLS_UPPER_COLOR = "#0077b6"

# "Base year" scenario group: DLS read on the "DLS ..." rows of the DLS sheet;
# for any other group (2050 scenarios), on "DLS 2050 ...".
DLS_BASE_YEAR_GROUP = "Base_year"

# BdF radial figure comparing 2 scenarios: central DLS of the base year
# (DLS_COLOR) and DLS 2050, in a slightly darker blue.
DLS_2050_COLOR = "#0096c7"

# (key, row suffix in the DLS sheet, color, legend label). A missing row (e.g.
# no "DLS lower bound" row in thresholds.xlsx/DLS) is simply not drawn.
DLS_LEVELS = (
    ("dls_lower", " lower bound", DLS_LOWER_COLOR, "DLS lower bound"),
    ("dls", "", DLS_COLOR, "DLS"),
    ("dls_upper", " upper bound", DLS_UPPER_COLOR, "DLS upper bound"),
)

# Name (case-insensitive) of the trend scenario: "TEND" or "TEND_2050".
# Placed right next to the reference in the figures that sort the scenarios
# (processing.sort_scenarios_by_reduction). Matched on the WHOLE name: the
# variants ("S1-TEND", "TEND-TEND", "TEND_Sus"...) are sorted with the other
# scenarios.
TREND_SCENARIO_MARKER = "TEND"
TREND_SCENARIO_PATTERN = rf"^{TREND_SCENARIO_MARKER}(_\d{{4}})?$"

# ============================================================================
# OVERSHOOT / SAFE OPERATING SPACE FIGURE
# ============================================================================

# Relative bounds (in multiples of the lower limit) of the safe/risk background gradient.
SAFE_LIMIT_REL = 1.0
RISK_TRANSITION_REL = 2.0
