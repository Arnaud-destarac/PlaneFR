"""
Bubble chart by indicator: for each consumption category (Food, Housing,
Mobility, Final goods, Final services) or each sector, 2 vertically stacked
bubbles -- M_dom+M_imp (French products, size Y_dom) and M_row (imported
products, size Y_imp).

Aggregation by WEIGHTED MEAN: for each category, each product flagged in the
'M' bridge contributes to M_category weighted by the Y of the matching category
(Housing -> Y_Residential, Mobility -> Y_Transport, otherwise positional
matching) -- see build_bridge_to_y_mapping().

Two-step pipeline:
  1. export_indicator_tables_excel: 1 workbook per scenario, 1 sheet per
     indicator, with a block of rows for the categories (on top) and another
     for the sectors (rows = x_dom/S_dom/F_x_dom/M_imp/M_dom/M_row/Y_imp/Y_dom and
     M*Y products), each sorted according to the row chosen in B1 (editable
     by hand in Excel).
  2. create_driver_comparison: bubble chart read back from these workbooks, by
     first n consumption categories or first n sectors (type="categories"/"products"),
     baseline alone or compared with a 2nd scenario (scenarios=False/True).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, figure_export, io, processing

# Scaling of the bubble sizes.
BUBBLE_SCALE = 1600.0
BUBBLE_MIN_SIZE = 30.0

# Plotted series: (name, rows summed for the ordinate, row giving the size).
# M_dom and M_imp are merged: both apply to Y_dom (French products), and for
# the categories their sum remains a mean weighted by Y_dom (sum of means
# weighted by the same weights).
BUBBLE_SERIES = [
    ("M_dom+M_imp", ("M_dom", "M_imp"), "Y_dom"),
    ("M_row", ("M_row",), "Y_imp"),
]

BUBBLE_COLORS = {
    "M_dom+M_imp": "#2ca02c",
    "M_row": "#ff7f0e",
}

# Colors of the vertical axes in comparison mode (left: baseline, right: scenario).
COMPARISON_AXIS_COLORS = ("#1f77b4", "#ff7f0e")


def build_bridge_to_y_mapping(bridge_cols, y_cols):
    """Map each column of the M bridge to a column of Y for the weighting.

    Special rules: Housing -> Residential, Mobility -> Transport, Food -> Food.
    Otherwise, positional matching (i-th bridge column -> i-th Y column).
    """
    mapping = {}
    y_cols_l = [str(c).lower() for c in y_cols]

    def find_y_col(preferred_tokens, fallback_idx):
        for tok in preferred_tokens:
            for i, yc in enumerate(y_cols_l):
                if tok in yc:
                    return y_cols[i]
        return y_cols[min(fallback_idx, len(y_cols) - 1)]

    for i, bcol in enumerate(bridge_cols):
        b = str(bcol).lower()
        if "housing" in b:
            mapping[bcol] = find_y_col(["residential", "housing"], i)
        elif "mobility" in b or "transport" in b:
            mapping[bcol] = find_y_col(["transport", "mobility"], i)
        elif "food" in b:
            mapping[bcol] = find_y_col(["food", "nour"], i)
        else:
            mapping[bcol] = y_cols[min(i, len(y_cols) - 1)]
    return mapping


def align_series_to_bridge(vector, bridge_m):
    """Align a vector (200 products) on the index of bridge_m, robustly."""
    vec = vector.copy()
    if len(vec) == len(bridge_m):
        vec.index = bridge_m.index
    else:
        vec = vec.reindex(bridge_m.index)
    return vec.fillna(0.0)


def weighted_mean_by_category(m_vector, y_5cols, bridge_m, bridge_to_y_col):
    """M aggregated by category via a mean weighted by Y:
    M_c = sum_i( M_i * Y_i,col(c) * I(i in c) ) / sum_i( Y_i,col(c) * I(i in c) )
    """
    m_vec = align_series_to_bridge(m_vector, bridge_m)
    y_aligned = y_5cols.copy()
    if len(y_aligned) == len(bridge_m):
        y_aligned.index = bridge_m.index
    else:
        y_aligned = y_aligned.reindex(bridge_m.index)
    y_aligned = y_aligned.fillna(0.0)

    out = {}
    for cat in bridge_m.columns:
        mask = bridge_m[cat] == 1
        y_col = bridge_to_y_col[cat]
        weights = y_aligned.loc[mask, y_col].astype(float)
        values = m_vec.loc[mask].astype(float)
        denom = float(weights.sum())
        out[cat] = float((values * weights).sum() / denom) if denom > 0 else 0.0
    return pd.Series(out)


def aggregate_y_for_sizes(y_5cols, bridge_m, bridge_to_y_col):
    """Y by category (sum), to size the bubbles."""
    y_aligned = y_5cols.copy()
    if len(y_aligned) == len(bridge_m):
        y_aligned.index = bridge_m.index
    else:
        y_aligned = y_aligned.reindex(bridge_m.index)
    y_aligned = y_aligned.fillna(0.0)

    out = {}
    for cat in bridge_m.columns:
        mask = bridge_m[cat] == 1
        y_col = bridge_to_y_col[cat]
        out[cat] = float(y_aligned.loc[mask, y_col].sum())
    return pd.Series(out)


def load_x_dom(scenario_folder_path):
    """Domestic production x_dom: first 200 rows of system/x.pkl (Series, system order)."""
    return pd.read_pickle(Path(scenario_folder_path) / "system" / "x.pkl").iloc[:200, 0]


def _load_s_dom(lp_dir, x_dom):
    """Direct intensity of the domestic sectors of an LP (rows = stressors,
    columns = the 200 domestic sectors, same format as S_x_dom.pkl):
      - S_x_dom.pkl if present;
      - otherwise, F_x_dom / x_dom when F_x_dom is by sector (200 columns:
        all the extensions since reformat_d_cba_k.py writes ghg_combustion and
        raw_materials in this format);
      - otherwise (data reformatted by an earlier version):
      - otherwise, extensions in product format (F_Z.pkl 400 x 200 and
        F_Y.pkl 400 x 6, rows = products burned / extracted, columns = using
        sectors / final demand). Stressors = rows of S_Z.pkl:
          * a single one (ghg_combustion: CO2) -> emissions of the sector that
            BURNS the fuel: S[j] = sum over the 400 rows (domestic + imported
            fuels) of column j of F_Z / x_j. F_Y (combustion by households)
            belongs to no sector and is not included, so S.x != F_x_dom;
          * one per product (raw_materials: material extracted) -> extraction
            attributed to the extracting sector: F[i] = sum of the domestic row
            i of F_Z + F_Y (= F_x_dom[i] exactly), S[i, i] = F[i] / x_i, row
            "product i" non-zero only in sector i (totals such as "RMC" dropped).
    0 where x_dom = 0. None if no usable file."""
    s_path, f_path = lp_dir / "S_x_dom.pkl", lp_dir / "F_x_dom.pkl"
    if s_path.exists():
        return pd.read_pickle(s_path)
    x = x_dom.to_numpy(dtype=float)
    inv_x = np.divide(1.0, x, out=np.zeros_like(x), where=x != 0)
    if f_path.exists():
        f_df = pd.read_pickle(f_path)
        if f_df.shape[1] == len(x_dom):
            return f_df * inv_x  # positional: columns of F = rows of x

    fz_path, fy_path, sz_path = lp_dir / "F_Z.pkl", lp_dir / "F_Y.pkl", lp_dir / "S_Z.pkl"
    if not (fz_path.exists() and sz_path.exists()):
        return None
    f_z = pd.read_pickle(fz_path)
    f_y = pd.read_pickle(fy_path) if fy_path.exists() else None

    def domestic_rows(df):
        if isinstance(df.index, pd.MultiIndex) and "origin" in df.index.names:
            return df[df.index.get_level_values("origin") == "domestic"]
        return df.iloc[:len(x)]

    columns = x_dom.index.droplevel(0) if isinstance(x_dom.index, pd.MultiIndex) else x_dom.index
    stressors = [s[0] if isinstance(s, tuple) else s for s in pd.read_pickle(sz_path).index]
    if len(stressors) == 1:
        # Sector that burns the fuel: column sums of F_Z (columns = the 200 domestic sectors)
        burned = f_z.sum(axis=0).to_numpy(dtype=float)
        return pd.DataFrame([burned * inv_x], index=stressors, columns=columns)

    # Extracting sector: domestic row sums of F_Z + F_Y
    f_by_sector = domestic_rows(f_z).sum(axis=1).to_numpy(dtype=float)
    if f_y is not None:
        f_by_sector = f_by_sector + domestic_rows(f_y).sum(axis=1).to_numpy(dtype=float)
    s_by_sector = f_by_sector * inv_x
    sector_names = list(columns.get_level_values(-1))
    matched = [name for name in stressors if name in sector_names]  # drops totals such as "RMC"
    s = np.zeros((len(matched), len(x)))
    for i, name in enumerate(matched):
        j = sector_names.index(name)
        s[i, j] = s_by_sector[j]
    return pd.DataFrame(s, index=matched, columns=columns)


def compute_indicator_m_sector_vectors(scenario_folder_path, cf_df):
    """M_dom/M_imp/M_row (and S_dom: direct intensity dom_{lp}/S_x_dom.pkl) by
    sector (200 products), aggregated by indicator (sum over all the associated
    LPs, weighted by the characterization factors).

    Returns:
        {name: {"M_dom", "M_imp", "M_row", "S_dom"}} -- Series indexed by sector.
    """
    indicator_to_lp = processing.get_unique_indicators(cf_df)
    x_dom = load_x_dom(scenario_folder_path)

    sector_vectors = {}
    for indicator_name, lp_list in indicator_to_lp.items():
        agg_dom = agg_imp = agg_row = None
        agg_s = None
        extension_factor_map = processing.build_extension_factor_map(cf_df, indicator_name)

        for lp_name in lp_list:
            lp_dir = Path(scenario_folder_path) / "extensions" / f"dom_{lp_name}"
            s_df = _load_s_dom(lp_dir, x_dom)
            if s_df is not None:
                v_s = processing.filter_and_weight_vector(s_df, extension_factor_map)
                # Start from the first vector (not an empty Series): adding to
                # an empty Series sorts the index alphabetically, which breaks
                # the positional alignment on the bridge (align_series_to_bridge).
                agg_s = v_s.copy() if agg_s is None else agg_s.add(v_s, fill_value=0)
            m_dom_df = io.load_m_matrix(scenario_folder_path, lp_name, "dom")
            m_imp_df = io.load_m_matrix(scenario_folder_path, lp_name, "imp")
            m_row_df = io.load_m_matrix(scenario_folder_path, lp_name, "row")
            if m_dom_df is None and m_imp_df is None and m_row_df is None:
                continue

            v_dom = processing.filter_and_weight_vector(m_dom_df, extension_factor_map) if m_dom_df is not None else pd.Series(dtype=float)
            v_imp = processing.filter_and_weight_vector(m_imp_df, extension_factor_map) if m_imp_df is not None else pd.Series(dtype=float)
            v_row = processing.filter_and_weight_vector(m_row_df, extension_factor_map) if m_row_df is not None else pd.Series(dtype=float)

            if agg_dom is None:
                agg_dom, agg_imp, agg_row = v_dom.copy(), v_imp.copy(), v_row.copy()
            else:
                agg_dom = agg_dom.add(v_dom, fill_value=0)
                agg_imp = agg_imp.add(v_imp, fill_value=0)
                agg_row = agg_row.add(v_row, fill_value=0)

        if agg_dom is None:
            continue

        sector_vectors[indicator_name] = {"M_dom": agg_dom, "M_imp": agg_imp, "M_row": agg_row,
                                         "S_dom": agg_s if agg_s is not None else pd.Series(dtype=float)}

    return sector_vectors


# ============================================================================
# EXCEL EXPORT
# ============================================================================

_EXCEL_SHEET_FORBIDDEN_CHARS = r'[]:*?/\\'


def _excel_sheet_name(name, used_names):
    """Valid (31 characters max, without []:*?/\\) and unique Excel sheet name."""
    clean = "".join("_" if ch in _EXCEL_SHEET_FORBIDDEN_CHARS else ch for ch in str(name)).strip() or "Sheet"
    candidate = clean[:31]
    suffix = 2
    while candidate.lower() in used_names:
        tag = f"_{suffix}"
        candidate = clean[:31 - len(tag)] + tag
        suffix += 1
    used_names.add(candidate.lower())
    return candidate


# Total footprint of an item: M_imp*Y_dom + M_dom*Y_dom + M_row*Y_imp (= d_cba,
# up to the exports). TOTAL_FOOTPRINT_ROW_LEGACY: label of this row in the
# workbooks written before it was renamed, still accepted when reading back.
TOTAL_FOOTPRINT_ROW = "d_cba"
TOTAL_FOOTPRINT_ROW_LEGACY = "M_imp*Y_dom+M_dom*Y_dom+M_row*Y_imp"

# Labels of column A delimiting the "source data" block of a sheet: written by
# _write_sortable_sheet, read back by load_indicator_tables_excel. The
# *_LEGACY labels are those of the workbooks written by an earlier (French)
# version of this module, still accepted when reading back.
SOURCE_BLOCK_TITLE = "Source data (bridge order, unsorted)"
KEY_ROW_LABEL = "Sort key"
RANK_ROW_LABEL = "Rank"  # rank of the sectors; also used to count the sectors when reading back
CATEGORY_RANK_ROW_LABEL = "Rank (categories)"
CATEGORY_KEY_ROW_LABEL = "Sort key (categories)"  # present only in the layout with separate blocks
SECTOR_SOURCE_HEADER = "Sectors"  # column A of the header of the sectors source block
SOURCE_BLOCK_TITLE_LEGACY = "Données sources (ordre du bridge, non triées)"
KEY_ROW_LABEL_LEGACY = "Clé de tri"
RANK_ROW_LABEL_LEGACY = "Rang"
CATEGORY_RANK_ROW_LABEL_LEGACY = "Rang (catégories)"


# Categories of Y.pkl excluded from the final demand of the Excel tables: GFCF
# (investment is already endogenized in M_k) and Exports.
EXCLUDED_Y_CATEGORIES = ("GFCF", "Exports")

# Outlier sectors excluded from the Excel tables (sector columns and aggregates
# by category): aberrant M_row intensities (RoW/dom ratio from 50 to ~30,000)
# that dominate GHG and PDF.
EXCLUDED_SECTORS = (
    "Private households with employed persons (95)",
    "Rubber and plastic products (25)",
)


def build_indicator_tables(scenario_folder_path, cf_df, bridge_m):
    """Table by indicator: 1 column per sector (200 products of the M bridge)
    + 1 column per consumption category, 1 row per data item (x_dom, S_dom,
    F_x_dom, M_imp, M_dom, M_row, Y_imp, Y_dom, then the M*Y products and their
    sums).

    Category columns: M aggregated by a mean weighted by Y (same computation as
    the bubble chart), Y and x_dom aggregated by sum, S_dom by a mean weighted
    by x_dom. F_x_dom = S_dom * x_dom (characterized direct pressure of the
    domestic production). The M*Y products and F_x_dom of a category are
    therefore exactly the sum of those of its sectors.

    Sources: x_dom = first 200 rows of system/x.pkl, S_dom = dom_{lp}/S_x_dom.pkl
    or F_x_dom / x_dom (see _load_s_dom), M_dom = dom_{lp}/M_k.pkl, M_imp = imp_{lp}/M_k.pkl, M_row =
    imp_{lp}/M_RoW.pkl, Y = system/Y.pkl without GFCF or Exports (Y_dom = first
    200 rows, Y_imp = next 200). Hence, by sector:
        M_dom*Y_dom             = dom_{lp}/d_cba_k
        M_imp*Y_dom+M_row*Y_imp = imp_{lp}/d_cba_k
    (relation checked exactly on the files, up to the exports: d_cba_k includes
    them, these tables do not).

    The sectors of EXCLUDED_SECTORS are removed from the sector columns and from
    the aggregates by category (rows set to 0 in the bridge, which keeps the
    positional alignment on the 200 products).

    Returns:
        {indicator_name: pd.DataFrame}
    """
    kept_sectors = ~bridge_m.index.isin(EXCLUDED_SECTORS)
    bridge_m = bridge_m.mul(kept_sectors, axis=0)
    kept_columns = np.concatenate([kept_sectors, np.ones(len(bridge_m.columns), dtype=bool)])

    y_dom_5, y_imp_5 = io.get_y_blocks(scenario_folder_path, excluded_y_categories=EXCLUDED_Y_CATEGORIES)
    bridge_to_y_col = build_bridge_to_y_mapping(list(bridge_m.columns), list(y_dom_5.columns))
    sector_vectors = compute_indicator_m_sector_vectors(scenario_folder_path, cf_df)

    y_dom_sector = align_series_to_bridge(y_dom_5.iloc[:, 0], bridge_m)
    y_imp_sector = align_series_to_bridge(y_imp_5.iloc[:, 0], bridge_m)
    y_dom_cat = aggregate_y_for_sizes(y_dom_5, bridge_m, bridge_to_y_col)
    y_imp_cat = aggregate_y_for_sizes(y_imp_5, bridge_m, bridge_to_y_col)

    # Domestic production x_dom (first 200 rows of system/x.pkl), in the same
    # 5-identical-columns format as Y: summed by category, weight of S_dom.
    x_dom = load_x_dom(scenario_folder_path)
    x_dom_5 = pd.concat([x_dom] * len(y_dom_5.columns), axis=1)
    x_dom_5.columns = y_dom_5.columns
    x_dom_sector = align_series_to_bridge(x_dom, bridge_m)
    x_dom_cat = aggregate_y_for_sizes(x_dom_5, bridge_m, bridge_to_y_col)

    tables = {}
    for indicator_name, vecs in sector_vectors.items():
        rows = {
            "x_dom": (x_dom_sector, x_dom_cat),
            "S_dom": (align_series_to_bridge(vecs["S_dom"], bridge_m),
                      weighted_mean_by_category(vecs["S_dom"], x_dom_5, bridge_m, bridge_to_y_col)),
            "M_imp": (align_series_to_bridge(vecs["M_imp"], bridge_m),
                      weighted_mean_by_category(vecs["M_imp"], y_dom_5, bridge_m, bridge_to_y_col)),
            "M_dom": (align_series_to_bridge(vecs["M_dom"], bridge_m),
                      weighted_mean_by_category(vecs["M_dom"], y_dom_5, bridge_m, bridge_to_y_col)),
            "M_row": (align_series_to_bridge(vecs["M_row"], bridge_m),
                      weighted_mean_by_category(vecs["M_row"], y_imp_5, bridge_m, bridge_to_y_col)),
            "Y_imp": (y_imp_sector, y_imp_cat),
            "Y_dom": (y_dom_sector, y_dom_cat),
        }
        table = pd.DataFrame(
            {key: pd.concat([sector, cat]) for key, (sector, cat) in rows.items()}
        ).T
        # Direct pressure of domestic production (characterized F_x_dom), right
        # after S_dom: for a category, = sum over its sectors (S_dom weighted by x_dom)
        f_x_dom = table.loc["S_dom"] * table.loc["x_dom"]
        position = table.index.get_loc("S_dom") + 1
        table = pd.concat([table.iloc[:position], f_x_dom.rename("F_x_dom").to_frame().T, table.iloc[position:]])
        table.loc["M_imp*Y_dom"] = table.loc["M_imp"] * table.loc["Y_dom"]
        table.loc["M_dom*Y_dom"] = table.loc["M_dom"] * table.loc["Y_dom"]
        table.loc["M_imp*Y_dom+M_dom*Y_dom"] = table.loc["M_imp*Y_dom"] + table.loc["M_dom*Y_dom"]
        table.loc["M_row*Y_imp"] = table.loc["M_row"] * table.loc["Y_imp"]
        table.loc[TOTAL_FOOTPRINT_ROW] = table.loc["M_imp*Y_dom+M_dom*Y_dom"] + table.loc["M_row*Y_imp"]
        tables[indicator_name] = table.loc[:, kept_columns]
    return tables


def _write_sortable_sheet(ws, indicator_name, table, n_sectors, sort_row):
    """Write a sheet whose columns are sorted by Excel formulas, in decreasing
    order of the row named in B1 (drop-down list, editable by hand). The
    consumption categories and the sectors are in two separate blocks of rows
    (same row labels M_imp, M_dom..., columns starting at B in both), the
    categories on top. Layout:
      - row 1        : choice of the sort row (B1);
      - sorted tables (formulas): categories, then sectors, each followed by a
        row giving the share (%) of the sort indicator within its block;
      - source data (values, bridge order): categories block (followed by the
        "Sort key (categories)" and "Rank (categories)" rows), then sectors
        block (header "Sectors", followed by "Sort key" and "Rank").
    Rank = RANK + COUNTIF to break ties (sectors at 0 in particular), so that
    each position 1..n matches exactly one column.
    """
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    bold = Font(bold=True)
    red = Font(bold=True, color="C00000")
    labels = list(table.index)
    n_rows = len(labels)
    categories = table.iloc[:, n_sectors:]
    sectors = table.iloc[:, :n_sectors]

    # Row positions: sorted blocks, then source blocks (each block = header +
    # n_rows data rows + 1 extra row (share / sort key) [+ rank row]).
    sorted_cat_header = 3
    sorted_sec_header = sorted_cat_header + n_rows + 3
    src_title = sorted_sec_header + n_rows + 3
    src_cat_header = src_title + 1
    src_sec_header = src_cat_header + n_rows + 4

    ws["A1"] = "Sort categories and sectors by (decreasing order):"
    ws["A1"].font = bold
    ws["B1"] = sort_row
    ws["B1"].font = red
    label_range = f"$A${src_cat_header + 1}:$A${src_cat_header + n_rows}"
    validation = DataValidation(type="list", formula1=f"={label_range}", allow_blank=False)
    ws.add_data_validation(validation)
    validation.add("B1")
    ws.cell(src_title, 1, SOURCE_BLOCK_TITLE).font = bold

    blocks = [
        (categories, src_cat_header, indicator_name, sorted_cat_header,
         f"{indicator_name} - consumption categories", CATEGORY_KEY_ROW_LABEL, CATEGORY_RANK_ROW_LABEL),
        (sectors, src_sec_header, SECTOR_SOURCE_HEADER, sorted_sec_header,
         f"{indicator_name} - sectors", KEY_ROW_LABEL, RANK_ROW_LABEL),
    ]
    for block, src_header, src_label, sorted_header, sorted_label, key_label, rank_label in blocks:
        size = block.shape[1]
        src_first, src_last = src_header + 1, src_header + n_rows
        key_row, rank_row = src_last + 1, src_last + 2
        sorted_first = sorted_header + 1
        share_row = sorted_first + n_rows
        block_labels = f"$A${src_first}:$A${src_last}"

        # Source data (raw values, bridge order)
        ws.cell(src_header, 1, src_label).font = bold
        for j, name in enumerate(block.columns, start=2):
            ws.cell(src_header, j, str(name)).font = bold
        for i, label in enumerate(labels):
            ws.cell(src_first + i, 1, label).font = bold
            for j, value in enumerate(block.iloc[i].values, start=2):
                value = float(value)
                ws.cell(src_first + i, j, value if np.isfinite(value) else 0.0)
        ws.cell(key_row, 1, key_label).font = bold
        ws.cell(rank_row, 1, rank_label).font = bold

        # Sorted table headers
        ws.cell(sorted_header, 1, sorted_label).font = bold
        for i, label in enumerate(labels):
            ws.cell(sorted_first + i, 1, label).font = bold
        ws.cell(share_row, 1, '="Share of "&$B$1&" (%)"').font = red
        if size == 0:
            continue

        first, last = get_column_letter(2), get_column_letter(1 + size)
        keys = f"${first}${key_row}:${last}${key_row}"
        ranks = f"${first}${rank_row}:${last}${rank_row}"
        for j in range(2, 2 + size):
            c = get_column_letter(j)
            # Sort key: value of the row chosen in B1; rank within the block
            ws.cell(key_row, j, f"=INDEX({c}${src_first}:{c}${src_last},MATCH($B$1,{block_labels},0))")
            ws.cell(rank_row, j, f"=RANK({c}{key_row},{keys},0)"
                                 f"+COUNTIF(${first}${key_row}:{c}{key_row},{c}{key_row})-1")
        for k in range(size):
            j = 2 + k
            pos = f"MATCH({k + 1},{ranks},0)"
            ws.cell(sorted_header, j, f"=INDEX(${first}${src_header}:${last}${src_header},{pos})").font = bold
            for i in range(n_rows):
                r = src_first + i
                ws.cell(sorted_first + i, j, f"=INDEX(${first}${r}:${last}${r},{pos})")
            cell = ws.cell(share_row, j, f"=IFERROR(INDEX({keys},{pos})/SUM({keys})*100,0)")
            cell.number_format = "0.00"

    ws.column_dimensions["A"].width = 46
    ws.freeze_panes = "B2"


def export_indicator_tables_excel(scenario_folder_path, cf_df, bridge_m, output_path,
                                   sort_row=TOTAL_FOOTPRINT_ROW):
    """Write an Excel workbook with 1 sheet per indicator (see
    build_indicator_tables for the content, _write_sortable_sheet for the layout).

    The sectors, then the categories, are sorted in decreasing order of
    `sort_row` via formulas: changing cell B1 of a sheet in Excel (drop-down
    list) re-sorts this sheet. The sheet names are truncated to 31 characters
    (Excel limit); the full name of the indicator is recalled at the top of
    each table.

    Returns:
        Path of the written file.
    """
    from openpyxl import Workbook

    tables = build_indicator_tables(scenario_folder_path, cf_df, bridge_m)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    used_names = set()
    for indicator_name, table in tables.items():
        if sort_row not in table.index:
            raise ValueError(f"Unknown sort_row: {sort_row!r} (expected: {list(table.index)})")
        ws = wb.create_sheet(_excel_sheet_name(indicator_name, used_names))
        _write_sortable_sheet(ws, indicator_name, table, table.shape[1] - len(bridge_m.columns), sort_row)
    wb.calculation.fullCalcOnLoad = True
    wb.save(output_path)
    return output_path


def _index_of_any(values, labels, start=0):
    """Index of the first item of `values` (from `start`) equal to one of `labels`."""
    for i in range(start, len(values)):
        if values[i] in labels:
            return i
    raise ValueError(f"None of {labels} found")


def _read_source_block(rows, header_idx, end_idx):
    """(row labels, DataFrame) of a source block: header row at `header_idx`
    (column names from B), data rows until `end_idx` (excluded)."""
    header = rows[header_idx]
    n_cols = max((j for j, v in enumerate(header) if v is not None and j > 0), default=0)
    columns = [str(v) for v in header[1:n_cols + 1]]
    data_rows = rows[header_idx + 1:end_idx]
    values = np.array([[float(v) if v is not None else 0.0 for v in (list(r[1:n_cols + 1]) + [None] * n_cols)[:n_cols]]
                       for r in data_rows]).reshape(len(data_rows), n_cols)
    labels = [r[0] for r in data_rows]
    return labels, pd.DataFrame(values, index=labels, columns=columns)


def load_indicator_tables_excel(excel_path):
    """Read back a workbook written by export_indicator_tables_excel.

    Reads the "source data" block (values, no formulas) and the sort row chosen
    in B1 of each sheet -- possibly edited by hand in Excel. Does not depend on
    the recomputation of the formulas of the sorted table: the workbook can be
    read back whether or not it was opened/saved in Excel. Workbooks written by
    the earlier (French) version of this module are also accepted.

    Sectors and categories are returned in two separate DataFrames: some names
    exist in both (e.g. "Cattle", "Dairy products" with the M_split bridge), a
    selection by name on a common table would mix them up.

    Returns:
        {indicator_name: {"sector_table": DataFrame (rows = data, columns =
            sectors, bridge order), "category_table": DataFrame (columns =
            categories), "sort_row": str, "sectors": [...], "categories": [...]}}
    """
    from openpyxl import load_workbook

    source_titles = (SOURCE_BLOCK_TITLE, SOURCE_BLOCK_TITLE_LEGACY)
    key_labels = (KEY_ROW_LABEL, KEY_ROW_LABEL_LEGACY)
    rank_labels = (RANK_ROW_LABEL, RANK_ROW_LABEL_LEGACY)
    category_rank_labels = (CATEGORY_RANK_ROW_LABEL, CATEGORY_RANK_ROW_LABEL_LEGACY)

    wb = load_workbook(excel_path, read_only=True)
    sheets = {}
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        first_col = [r[0] if r else None for r in rows]
        if not any(title in first_col for title in source_titles):
            continue
        header_idx = _index_of_any(first_col, source_titles) + 1
        indicator_name = rows[header_idx][0]
        sort_row = rows[0][1]

        if CATEGORY_KEY_ROW_LABEL in first_col:
            # Current layout: categories block, then sectors block
            cat_key_idx = _index_of_any(first_col, (CATEGORY_KEY_ROW_LABEL,), header_idx)
            sec_header_idx = _index_of_any(first_col, (SECTOR_SOURCE_HEADER,), cat_key_idx)
            sec_key_idx = _index_of_any(first_col, key_labels, sec_header_idx)
            labels, category_table = _read_source_block(rows, header_idx, cat_key_idx)
            _, sector_table = _read_source_block(rows, sec_header_idx, sec_key_idx)
            up_to_date = True
        else:
            # Earlier layout: sectors then categories on the same rows
            key_idx = _index_of_any(first_col, key_labels, header_idx)
            rank_idx = _index_of_any(first_col, rank_labels, key_idx)
            labels, table = _read_source_block(rows, header_idx, key_idx)
            # The sectors are the ranked columns ("Rank" row filled in), the
            # categories those that follow.
            n_sectors = sum(v is not None for v in rows[rank_idx][1:table.shape[1] + 1])
            sector_table, category_table = table.iloc[:, :n_sectors], table.iloc[:, n_sectors:]
            up_to_date = False

        # Total footprint row renamed d_cba (former label in older workbooks)
        rename = {TOTAL_FOOTPRINT_ROW_LEGACY: TOTAL_FOOTPRINT_ROW}
        labels = [rename.get(label, label) for label in labels]
        sector_table, category_table = sector_table.rename(index=rename), category_table.rename(index=rename)
        sort_row = rename.get(sort_row, sort_row)

        if sort_row not in labels:
            raise ValueError(f"[{ws.title}] unknown sort row in B1: {sort_row!r} (expected: {labels})")
        sheets[indicator_name] = {
            "sector_table": sector_table,
            "category_table": category_table,
            "sort_row": sort_row,
            "sectors": list(sector_table.columns),
            "categories": list(category_table.columns),
            # False for a workbook in an earlier layout (to be regenerated)
            "categories_sorted": up_to_date,
        }
    wb.close()
    return sheets


BUBBLE_ROWS = ["M_imp", "M_dom", "M_row", "Y_imp", "Y_dom"]


def select_bubble_items(sheet, type="categories", n=None, sort_row=None):
    """Columns to plot for a sheet read back by load_indicator_tables_excel.

    type="categories": the first n consumption categories;
    type="products"  : the first n sectors.
    n=None: all of them. In both cases, columns in decreasing order of
    `sort_row` (None: the sort row of the sheet, B1), ties in bridge order
    (same rule as the RANK + COUNTIF of the workbook).

    Returns:
        DataFrame rows BUBBLE_ROWS x selected columns.
    """
    if type == "categories":
        table = sheet["category_table"]
    elif type == "products":
        table = sheet["sector_table"]
    else:
        raise ValueError(f"type must be 'categories' or 'products', got {type!r}")
    sort_row = sheet["sort_row"] if sort_row is None else sort_row
    if sort_row not in table.index:
        raise ValueError(f"Unknown sort_row: {sort_row!r} (expected: {list(table.index)})")
    order = np.argsort(-table.loc[sort_row].to_numpy(dtype=float), kind="stable")
    return table.iloc[:, order[:n]].loc[BUBBLE_ROWS]


def scale_bubble_sizes(values, scale=BUBBLE_SCALE, min_size=BUBBLE_MIN_SIZE, ref_max=None):
    """Normalize the bubble sizes to avoid giant bubbles, while keeping the
    proportion relative to the maximum (`ref_max`, by default the max of
    `values` -- to be given so that several subplots share the same scale)."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    ref = arr.max() if ref_max is None else ref_max
    if arr.size == 0 or ref <= 0:
        return np.full_like(arr, min_size, dtype=float)
    return min_size + (arr / ref) * scale


def _compute_axis_limits(series_list, default_max=1.0):
    """Y limits with a simple and robust margin for a set of series."""
    values = []
    for series in series_list:
        arr = np.asarray(series, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)

    if not values:
        return 0.0, default_max

    merged = np.concatenate(values)
    y_min, y_max = float(np.min(merged)), float(np.max(merged))

    if np.isclose(y_min, y_max):
        pad = 1.0 if np.isclose(y_max, 0.0) else abs(y_max) * 0.15
    else:
        pad = max((y_max - y_min) * 0.15, 1e-6)

    lower = min(0.0, y_min - 0.10 * pad)
    upper = y_max + pad
    if np.isclose(lower, upper):
        upper = lower + default_max
    return lower, upper


# ============================================================================
# FIGURES (from the Excel workbooks of export_indicator_tables_excel)
# ============================================================================

PRODUCT_LABEL_MAX_CHARS = 30


def _short_label(label, max_chars=PRODUCT_LABEL_MAX_CHARS):
    """Truncate the EXIOBASE sector names (often very long) for the X axis."""
    label = str(label)
    return label if len(label) <= max_chars else label[:max_chars - 1] + "…"


def _scenario_label(excel_path):
    """Scenario name deduced from the file name (bubble_data_by_indicator_<scenario>.xlsx)."""
    return Path(excel_path).stem.replace("bubble_data_by_indicator_", "")


def _series_values(payload, components):
    """Ordinate of a series of BUBBLE_SERIES: sum of the `components` rows."""
    return payload.loc[list(components)].to_numpy(dtype=float).sum(axis=0)


def _shares(sheet, items, type):
    """Share (%) of each displayed item in the total footprint of the
    indicator (TOTAL_FOOTPRINT_ROW, total = sum over all the sectors of the sheet)."""
    total = float(sheet["sector_table"].loc[TOTAL_FOOTPRINT_ROW].sum())
    if total == 0:
        return np.zeros(len(items))
    table = sheet["category_table"] if type == "categories" else sheet["sector_table"]
    values = table.loc[TOTAL_FOOTPRINT_ROW].reindex(items).fillna(0.0).to_numpy(dtype=float)
    return 100.0 * values / total


SHARE_BAR_COLOR = "#b0b0b0"
SHARE_AXIS_LABEL = "Share of total (%)"


def _style_share_axis(ax_bar, shares_list):
    """Left axis of the share (%) bars, below the bubbles."""
    top = max((float(np.max(s)) for s in shares_list if len(s)), default=0.0)
    ax_bar.set_ylim(0, top * 1.15 if top > 0 else 1.0)
    ax_bar.set_ylabel(SHARE_AXIS_LABEL, fontsize=9, color="#555555")
    ax_bar.tick_params(axis="y", labelsize=8, colors="#555555")
    ax_bar.spines["top"].set_visible(False)


def _ref_max(payloads, row):
    """Max of a row (Y_dom/Y_imp) over all the subplots: common bubble scale."""
    values = np.concatenate([p.loc[row].to_numpy(dtype=float) for p in payloads.values()])
    values = values[np.isfinite(values)]
    return float(values.max()) if values.size else 0.0


def _make_grid(n_subplots, n_items, type, base_width, base_height):
    n_cols = 3
    n_rows = int(math.ceil(n_subplots / n_cols))
    width = max(base_width, 0.6 * n_items + 2.0)
    height = base_height if type == "categories" else base_height + 1.0
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width * n_cols, height * n_rows))
    return fig, np.array(axes).reshape(-1)


def _style_x_axis(ax, x_pos, items, type):
    labels = items if type == "categories" else [_short_label(i) for i in items]
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8 if type == "categories" else 7)
    ax.set_xlim(-0.6, len(items) - 0.4)


def _figure_subject(type, n, sort_row):
    items = "consumption categories" if type == "categories" else "sectors"
    subject = f"{items} per indicator" if n is None else f"top {n} {items} per indicator"
    return f"{subject} (sorted by: {sort_row})"


def _common_sort_row(sheets, sort_row=None):
    """Sort row applied to all the indicators: `sort_row` if given, otherwise
    the B1 cell of the sheets, which must then be the same in all of them."""
    if sort_row is not None:
        return sort_row
    chosen = {name: sheet["sort_row"] for name, sheet in sheets.items()}
    if len(set(chosen.values())) > 1:
        details = "\n".join(f"  {name}: {row}" for name, row in chosen.items())
        raise ValueError("The sort row (B1) differs between the sheets of the workbook; the same one must "
                         "apply to all the indicators. Set the same B1 everywhere or pass sort_row=...\n"
                         + details)
    return next(iter(chosen.values()), TOTAL_FOOTPRINT_ROW)


def _bubble_rows(label, payloads, shares, sort_row, type):
    """Rows of the "Bubbles" sheet of create_driver_comparison for one
    scenario: one row per indicator x displayed item, in display order."""
    rows = []
    for indicator_name, payload in payloads.items():
        for rank, item in enumerate(payload.columns, start=1):
            column = payload[item]
            rows.append({
                "Scenario": label,
                "Indicator": indicator_name,
                "Sorted by": sort_row,
                "Rank": rank,
                "Consumption category" if type == "categories" else "Sector": item,
                **{f"{key} (bubble height)": figure_export.clean_value(column.loc[list(components)].sum())
                   for key, components, _ in BUBBLE_SERIES},
                **{row: figure_export.clean_value(column.loc[row]) for row in BUBBLE_ROWS},
                f"{SHARE_AXIS_LABEL} (bars)": figure_export.clean_value(shares[indicator_name][rank - 1]),
            })
    return rows


def create_driver_comparison(base_excel_path, type="categories", n=None, scenarios=False,
                             comparison_excel_path=None, base_label=None, s1_label=None,
                             excel_path=None, sort_row=None):
    """Bubble chart from the Excel workbooks of export_indicator_tables_excel:
    1 subplot per indicator, 2 bubbles per item of the X axis (see
    BUBBLE_SERIES): M_dom+M_imp (size Y_dom) and M_row (size Y_imp).

    Args:
        base_excel_path: workbook of the reference scenario (baseline).
        type: "categories" -> the first `n` consumption categories of each
            indicator; "products" -> the first `n` sectors of each indicator.
            In both cases, in decreasing order of the sort row (see sort_row),
            the same for all the indicators, recalled in the figure title.
        n: number of categories / sectors displayed (None: all).
        sort_row: row of the tables used to sort the items of every indicator
            (e.g. "d_cba", "M_row", "Y_dom"...). None: the row chosen in B1 of
            the baseline workbook (editable by hand in Excel), which must then
            be the same in all its sheets (ValueError otherwise).
        scenarios: False -> baseline alone; True -> comparison baseline vs
            `comparison_excel_path` (independent vertical axis per scenario).
            The displayed columns and their order are those of the baseline
            ranking, to compare the same items in both scenarios.
        base_label, s1_label: legends; by default deduced from the file names.
        excel_path: if given, Excel workbook of the plotted data (see
            figure_export), "Bubbles" sheet: one row per scenario x indicator x
            displayed item (in display order) with the bubble heights
            (M_dom+M_imp, M_row), the rows M_imp/M_dom/M_row/Y_imp/Y_dom (the
            bubble sizes are Y_dom and Y_imp) and the share of the item in the
            total footprint (bars).

    Returns:
        (fig, axes)
    """
    base_sheets = load_indicator_tables_excel(base_excel_path)
    base_label = base_label or _scenario_label(base_excel_path)
    sort_row = _common_sort_row(base_sheets, sort_row)
    base_payloads = {sp: select_bubble_items(sheet, type, n, sort_row) for sp, sheet in base_sheets.items()}
    base_shares = {sp: _shares(base_sheets[sp], list(p.columns), type) for sp, p in base_payloads.items()}
    titles = {sp: sp for sp in base_sheets}
    bubble_rows = _bubble_rows(base_label, base_payloads, base_shares, sort_row, type)

    if not scenarios:
        fig, axes = _plot_single(base_payloads, base_shares, titles, type, n, sort_row, base_label)
        subject = base_label
    else:
        if comparison_excel_path is None:
            raise ValueError("scenarios=True requires comparison_excel_path")
        s1_sheets = load_indicator_tables_excel(comparison_excel_path)
        s1_label = s1_label or _scenario_label(comparison_excel_path)
        s1_payloads, s1_shares = {}, {}
        for sp, base_payload in base_payloads.items():
            items = list(base_payload.columns)
            if sp in s1_sheets:
                table_key = "category_table" if type == "categories" else "sector_table"
                s1_payloads[sp] = s1_sheets[sp][table_key].reindex(index=BUBBLE_ROWS, columns=items).fillna(0.0)
                s1_shares[sp] = _shares(s1_sheets[sp], items, type)
            else:  # indicator missing from the compared scenario: bubbles and bars drawn at 0
                s1_payloads[sp] = pd.DataFrame(0.0, index=BUBBLE_ROWS, columns=items)
                s1_shares[sp] = np.zeros(len(items))
        fig, axes = _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                                     sort_row, base_label, s1_label)
        bubble_rows += _bubble_rows(s1_label, s1_payloads, s1_shares, sort_row, type)
        subject = f"{base_label} vs {s1_label}"

    if excel_path is not None and fig is not None:
        figure_export.write_figure_excel(excel_path, {"Bubbles": bubble_rows}, about={
            "Figure": f"Bubble chart - {subject} - {_figure_subject(type, n, sort_row)}",
            "Generated by": "planefr_lib.plot_bubble.create_driver_comparison",
            "Content": "Environmental intensities (multipliers M) and final demands (Y) of the displayed "
                       "consumption categories or sectors, by indicator. Bubble heights: M_dom+M_imp "
                       "(French products, size Y_dom) and M_row (imported products, size Y_imp).",
            "Source workbooks": ", ".join(Path(p).name for p in
                                          [base_excel_path] + ([comparison_excel_path] if scenarios else [])),
            "Order": f"Items in decreasing order of the {sort_row} row, the same for all the indicators "
                     "(the order of the reference scenario in comparison mode).",
            "Share": "Share of the item in the total footprint of the indicator (sum over all the sectors).",
        })
    return fig, axes


def _plot_single(payloads, shares, titles, type, n, sort_row, label):
    """1 subplot per indicator, 1 scenario: share (%) bars in the total on
    the left axis, bubbles on the right axis. Bubble scale common to all the
    subplots (max of Y_dom, resp. Y_imp, over the displayed items)."""
    indicators = list(payloads)
    if not indicators:
        print("No data to plot.")
        return None, None

    n_items = max(p.shape[1] for p in payloads.values())
    fig, axes = _make_grid(len(indicators), n_items, type, base_width=6.2, base_height=4.8)
    refs = {row: _ref_max(payloads, row) for row in ("Y_dom", "Y_imp")}

    for i, sp in enumerate(indicators):
        ax_bar = axes[i]
        ax = ax_bar.twinx()
        payload = payloads[sp]
        x_pos = np.arange(payload.shape[1])
        ax_bar.bar(x_pos, shares[sp], width=0.6, color=SHARE_BAR_COLOR, alpha=0.5)
        _style_share_axis(ax_bar, [shares[sp]])

        for key, components, size_row in BUBBLE_SERIES:
            sizes = scale_bubble_sizes(payload.loc[size_row], ref_max=refs[size_row])
            ax.scatter(x_pos, _series_values(payload, components), s=sizes, c=BUBBLE_COLORS[key],
                       alpha=0.6, edgecolors="none")
        ax.set_ylim(*_compute_axis_limits([_series_values(payload, c) for _, c, _ in BUBBLE_SERIES]))
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylabel("Intensity M (bubbles)", fontsize=9)
        ax.spines["top"].set_visible(False)

        ax_bar.set_title(titles[sp], fontsize=11 if type == "categories" else 10, fontweight="bold", pad=8)
        _style_x_axis(ax_bar, x_pos, list(payload.columns), type)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for i in range(len(indicators), len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUBBLE_COLORS[key], markersize=10,
               alpha=0.7, label=f"{key} (size: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [Patch(facecolor=SHARE_BAR_COLOR, alpha=0.5, label=f"{SHARE_AXIS_LABEL} (left axis)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Bubble chart - {label} - {_figure_subject(type, n, sort_row)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                     sort_row, base_label, s1_label):
    """Comparison of two scenarios: share (%) bars in the total of each scenario
    on the left axis; bubbles on two independent right axes (base_label against
    the frame, s1_label shifted outwards), shifted horizontally. Bubble scale
    specific to each scenario, common to all the subplots."""
    indicators = list(base_payloads)
    if not indicators:
        print("No data to plot.")
        return None, None

    n_items = max(p.shape[1] for p in base_payloads.values())
    fig, axes = _make_grid(len(indicators), n_items, type, base_width=7.4, base_height=5.1)
    offset = 0.14

    base_colors = BUBBLE_COLORS
    s1_colors = {
        "M_dom+M_imp": colors.get_light_shade(base_colors["M_dom+M_imp"], factor=1.20),
        "M_row": colors.get_light_shade(base_colors["M_row"], factor=1.18),
    }
    left_color, right_color = COMPARISON_AXIS_COLORS
    base_refs = {row: _ref_max(base_payloads, row) for row in ("Y_dom", "Y_imp")}
    s1_refs = {row: _ref_max(s1_payloads, row) for row in ("Y_dom", "Y_imp")}

    base_bar_color, s1_bar_color = left_color, right_color

    for i, sp in enumerate(indicators):
        ax_bar = axes[i]
        ax_base = ax_bar.twinx()
        ax_s1 = ax_bar.twinx()
        ax_s1.spines["right"].set_position(("axes", 1.14))

        base, s1 = base_payloads[sp], s1_payloads[sp]
        x_pos = np.arange(base.shape[1])

        ax_bar.bar(x_pos - offset, base_shares[sp], width=2 * offset, color=base_bar_color, alpha=0.25)
        ax_bar.bar(x_pos + offset, s1_shares[sp], width=2 * offset, color=s1_bar_color, alpha=0.25)
        _style_share_axis(ax_bar, [base_shares[sp], s1_shares[sp]])

        for key, components, size_row in BUBBLE_SERIES:
            ax_base.scatter(x_pos - offset, _series_values(base, components),
                            s=scale_bubble_sizes(base.loc[size_row], ref_max=base_refs[size_row]),
                            c=base_colors[key], alpha=0.72, edgecolors="none")
            ax_s1.scatter(x_pos + offset, _series_values(s1, components),
                          s=scale_bubble_sizes(s1.loc[size_row], ref_max=s1_refs[size_row]),
                          c=s1_colors[key], alpha=0.55, edgecolors=base_colors[key], linewidths=1.0)

        ax_base.set_ylim(*_compute_axis_limits([_series_values(base, c) for _, c, _ in BUBBLE_SERIES]))
        ax_s1.set_ylim(*_compute_axis_limits([_series_values(s1, c) for _, c, _ in BUBBLE_SERIES]))

        ax_bar.set_title(f"{titles[sp]}\n{base_label} vs {s1_label}",
                         fontsize=11 if type == "categories" else 10, fontweight="bold", pad=8)
        _style_x_axis(ax_bar, x_pos, list(base.columns), type)
        ax_base.set_ylabel(base_label, fontsize=9, fontweight="bold", color=left_color)
        ax_s1.set_ylabel(s1_label, fontsize=9, fontweight="bold", color=right_color)
        ax_base.tick_params(axis="y", labelsize=8, colors=left_color)
        ax_s1.tick_params(axis="y", labelsize=8, colors=right_color)
        ax_base.spines["right"].set_color(left_color)
        ax_s1.spines["right"].set_color(right_color)
        ax_base.grid(axis="y", alpha=0.22, linestyle="--")
        ax_base.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)

        for ax in (ax_base, ax_s1):
            ax.spines["top"].set_visible(False)
            ax.spines["left"].set_visible(False)

    for i in range(len(indicators), len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=base_colors[key], markeredgecolor="none",
               markersize=9, label=f"{base_label} - {key} (size: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=s1_colors[key], markeredgecolor=base_colors[key],
               markersize=9, label=f"{s1_label} - {key} (size: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [
        Patch(facecolor=base_bar_color, alpha=0.25, label=f"{base_label} - {SHARE_AXIS_LABEL} (left axis)"),
        Patch(facecolor=s1_bar_color, alpha=0.25, label=f"{s1_label} - {SHARE_AXIS_LABEL} (left axis)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True, fancybox=True, shadow=True)
    fig.suptitle(f"Bubble chart - {base_label} vs {s1_label} - {_figure_subject(type, n, sort_row)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig, axes
