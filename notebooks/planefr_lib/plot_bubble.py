"""
Bubble chart by sub-process: for each consumption category (Food, Housing,
Mobility, Final goods, Final services) or each sector, 2 vertically stacked
bubbles -- M_dom+M_imp (French products, size Y_dom) and M_row (imported
products, size Y_imp).

Aggregation by WEIGHTED MEAN: for each category, each product flagged in the
'M' bridge contributes to M_category weighted by the Y of the matching category
(Housing -> Y_Residential, Mobility -> Y_Transport, otherwise positional
matching) -- see build_bridge_to_y_mapping().

Two-step pipeline:
  1. export_subprocess_tables_excel: 1 workbook per scenario, 1 sheet per
     sub-process (columns = sectors + categories, rows = M_imp/M_dom/
     M_row/Y_imp/Y_dom and M*Y products), sectors sorted according to the row
     chosen in B1 (editable by hand in Excel).
  2. plot_bubble_chart: bubble chart read back from these workbooks, by
     consumption categories or by first n sectors (type="categories"/"products"),
     baseline alone or compared with a 2nd scenario (scenarios=False/True).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, io, processing

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


def compute_subprocess_m_sector_vectors(scenario_folder_path, cf_df):
    """M_dom/M_imp/M_row by sector (200 products), aggregated by sub-process
    (sum over all the associated LPs, weighted by the characterization factors).

    Returns:
        {name: {"M_dom", "M_imp", "M_row"}} -- Series indexed by sector.
    """
    subprocess_to_lp = processing.get_unique_subprocesses(cf_df)

    sector_vectors = {}
    for subprocess_name, lp_list in subprocess_to_lp.items():
        agg_dom = agg_imp = agg_row = None
        extension_factor_map = processing.build_extension_factor_map(cf_df, subprocess_name)

        for lp_name in lp_list:
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

        sector_vectors[subprocess_name] = {"M_dom": agg_dom, "M_imp": agg_imp, "M_row": agg_row}

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


TOTAL_FOOTPRINT_ROW = "M_imp*Y_dom+M_dom*Y_dom+M_row*Y_imp"

# Labels of column A delimiting the "source data" block of a sheet: written by
# _write_sortable_sheet, read back by load_subprocess_tables_excel. The
# *_LEGACY labels are those of the workbooks written by an earlier (French)
# version of this module, still accepted when reading back.
SOURCE_BLOCK_TITLE = "Source data (bridge order, unsorted)"
KEY_ROW_LABEL = "Sort key"
RANK_ROW_LABEL = "Rank"  # rank of the sectors; also used to count the sectors when reading back
CATEGORY_RANK_ROW_LABEL = "Rank (categories)"
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


def build_subprocess_tables(scenario_folder_path, cf_df, bridge_m):
    """Table by sub-process: 1 column per sector (200 products of the M bridge)
    + 1 column per consumption category, 1 row per data item (M_imp, M_dom,
    M_row, Y_imp, Y_dom, then the M*Y products and their sums).

    Category columns: M aggregated by a mean weighted by Y (same computation as
    the bubble chart), Y aggregated by sum. The M*Y products of a category are
    therefore exactly the sum of the M_i*Y_i of its sectors.

    Sources: M_dom = dom_{lp}/M_k.pkl, M_imp = imp_{lp}/M_k.pkl, M_row =
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
        {subprocess_name: pd.DataFrame}
    """
    kept_sectors = ~bridge_m.index.isin(EXCLUDED_SECTORS)
    bridge_m = bridge_m.mul(kept_sectors, axis=0)
    kept_columns = np.concatenate([kept_sectors, np.ones(len(bridge_m.columns), dtype=bool)])

    y_dom_5, y_imp_5 = io.get_y_blocks(scenario_folder_path, excluded_y_categories=EXCLUDED_Y_CATEGORIES)
    bridge_to_y_col = build_bridge_to_y_mapping(list(bridge_m.columns), list(y_dom_5.columns))
    sector_vectors = compute_subprocess_m_sector_vectors(scenario_folder_path, cf_df)

    y_dom_sector = align_series_to_bridge(y_dom_5.iloc[:, 0], bridge_m)
    y_imp_sector = align_series_to_bridge(y_imp_5.iloc[:, 0], bridge_m)
    y_dom_cat = aggregate_y_for_sizes(y_dom_5, bridge_m, bridge_to_y_col)
    y_imp_cat = aggregate_y_for_sizes(y_imp_5, bridge_m, bridge_to_y_col)

    tables = {}
    for subprocess_name, vecs in sector_vectors.items():
        rows = {
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
        table.loc["M_imp*Y_dom"] = table.loc["M_imp"] * table.loc["Y_dom"]
        table.loc["M_dom*Y_dom"] = table.loc["M_dom"] * table.loc["Y_dom"]
        table.loc["M_imp*Y_dom+M_dom*Y_dom"] = table.loc["M_imp*Y_dom"] + table.loc["M_dom*Y_dom"]
        table.loc["M_row*Y_imp"] = table.loc["M_row"] * table.loc["Y_imp"]
        table.loc[TOTAL_FOOTPRINT_ROW] = table.loc["M_imp*Y_dom+M_dom*Y_dom"] + table.loc["M_row*Y_imp"]
        tables[subprocess_name] = table.loc[:, kept_columns]
    return tables


def _write_sortable_sheet(ws, subprocess_name, table, n_sectors, sort_row):
    """Write a sheet whose columns are sorted by Excel formulas, in decreasing
    order of the row named in B1 (drop-down list, editable by hand): the
    sectors among themselves, then the categories among themselves. Layout:
      - row 1        : choice of the sort row (B1);
      - top block    : sorted table (formulas) -- sorted sectors, then sorted
                       categories, + a last row giving the share (%) of the
                       sort indicator within its group;
      - bottom block : source data (values, bridge order), followed by the
                       "Sort key", "Rank" (sectors) and "Rank (categories)"
                       computation rows.
    Rank = RANK + COUNTIF to break ties (sectors at 0 in particular), so that
    each position 1..n matches exactly one column.
    """
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation

    bold = Font(bold=True)
    labels = list(table.index)
    n_rows, n_cols = len(labels), table.shape[1]

    sorted_header = 3
    sorted_first = sorted_header + 1
    share_row = sorted_first + n_rows
    src_title = sorted_first + n_rows + 2
    src_header = src_title + 1
    src_first = src_header + 1
    src_last = src_first + n_rows - 1
    key_row = src_last + 1
    rank_row = key_row + 1
    category_rank_row = rank_row + 1
    label_range = f"$A${src_first}:$A${src_last}"

    ws["A1"] = "Sort sectors and categories by (decreasing order):"
    ws["A1"].font = bold
    ws["B1"] = sort_row
    ws["B1"].font = Font(bold=True, color="C00000")
    validation = DataValidation(type="list", formula1=f"={label_range}", allow_blank=False)
    ws.add_data_validation(validation)
    validation.add("B1")

    # Source data (raw values, bridge order)
    ws.cell(src_title, 1, SOURCE_BLOCK_TITLE).font = bold
    ws.cell(src_header, 1, subprocess_name).font = bold
    for j, name in enumerate(table.columns, start=2):
        ws.cell(src_header, j, str(name)).font = bold
    for i, label in enumerate(labels):
        ws.cell(src_first + i, 1, label).font = bold
        for j, value in enumerate(table.iloc[i].values, start=2):
            value = float(value)
            ws.cell(src_first + i, j, value if np.isfinite(value) else 0.0)

    # Sort key: value of the row chosen in B1, for each column
    ws.cell(key_row, 1, KEY_ROW_LABEL).font = bold
    for j in range(2, 2 + n_cols):
        c = get_column_letter(j)
        ws.cell(key_row, j, f"=INDEX({c}${src_first}:{c}${src_last},MATCH($B$1,{label_range},0))")

    ws.cell(sorted_header, 1, subprocess_name).font = bold
    for i, label in enumerate(labels):
        ws.cell(sorted_first + i, 1, label).font = bold
    ws.cell(share_row, 1, '="Share of "&$B$1&" (%)"').font = Font(bold=True, color="C00000")

    # Sorting of each group (sectors, then categories) independently: rank in
    # the group, columns of the sorted table reordered, share (%) in the total
    # of the group.
    groups = [
        (2, n_sectors, rank_row, RANK_ROW_LABEL),
        (2 + n_sectors, n_cols - n_sectors, category_rank_row, CATEGORY_RANK_ROW_LABEL),
    ]
    for start, size, group_rank_row, rank_label in groups:
        if size == 0:
            continue
        first, last = get_column_letter(start), get_column_letter(start + size - 1)
        keys = f"${first}${key_row}:${last}${key_row}"
        ranks = f"${first}${group_rank_row}:${last}${group_rank_row}"
        ws.cell(group_rank_row, 1, rank_label).font = bold
        for j in range(start, start + size):
            c = get_column_letter(j)
            ws.cell(group_rank_row, j, f"=RANK({c}{key_row},{keys},0)"
                                       f"+COUNTIF(${first}${key_row}:{c}{key_row},{c}{key_row})-1")
        for k in range(size):
            j = start + k
            pos = f"MATCH({k + 1},{ranks},0)"
            ws.cell(sorted_header, j, f"=INDEX(${first}${src_header}:${last}${src_header},{pos})").font = bold
            for i in range(n_rows):
                r = src_first + i
                ws.cell(sorted_first + i, j, f"=INDEX(${first}${r}:${last}${r},{pos})")
            cell = ws.cell(share_row, j, f"=IFERROR(INDEX({keys},{pos})/SUM({keys})*100,0)")
            cell.number_format = "0.00"

    ws.column_dimensions["A"].width = 46
    ws.freeze_panes = ws.cell(sorted_first, 2)


def export_subprocess_tables_excel(scenario_folder_path, cf_df, bridge_m, output_path,
                                   sort_row=TOTAL_FOOTPRINT_ROW):
    """Write an Excel workbook with 1 sheet per sub-process (see
    build_subprocess_tables for the content, _write_sortable_sheet for the layout).

    The sectors, then the categories, are sorted in decreasing order of
    `sort_row` via formulas: changing cell B1 of a sheet in Excel (drop-down
    list) re-sorts this sheet. The sheet names are truncated to 31 characters
    (Excel limit); the full name of the sub-process is recalled at the top of
    each table.

    Returns:
        Path of the written file.
    """
    from openpyxl import Workbook

    tables = build_subprocess_tables(scenario_folder_path, cf_df, bridge_m)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    used_names = set()
    for subprocess_name, table in tables.items():
        if sort_row not in table.index:
            raise ValueError(f"Unknown sort_row: {sort_row!r} (expected: {list(table.index)})")
        ws = wb.create_sheet(_excel_sheet_name(subprocess_name, used_names))
        _write_sortable_sheet(ws, subprocess_name, table, table.shape[1] - len(bridge_m.columns), sort_row)
    wb.calculation.fullCalcOnLoad = True
    wb.save(output_path)
    return output_path


def _index_of_any(values, labels, start=0):
    """Index of the first item of `values` (from `start`) equal to one of `labels`."""
    for i in range(start, len(values)):
        if values[i] in labels:
            return i
    raise ValueError(f"None of {labels} found")


def load_subprocess_tables_excel(excel_path):
    """Read back a workbook written by export_subprocess_tables_excel.

    Reads the "source data" block (values, no formulas) and the sort row chosen
    in B1 of each sheet -- possibly edited by hand in Excel. Does not depend on
    the recomputation of the formulas of the sorted table: the workbook can be
    read back whether or not it was opened/saved in Excel. Workbooks written by
    the earlier (French) version of this module are also accepted.

    Sectors and categories are returned in two separate DataFrames: some names
    exist in both (e.g. "Cattle", "Dairy products" with the M_split bridge), a
    selection by name on a common table would mix them up.

    Returns:
        {subprocess_name: {"sector_table": DataFrame (rows = data, columns =
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
        key_idx = _index_of_any(first_col, key_labels, header_idx)
        rank_idx = _index_of_any(first_col, rank_labels, key_idx)

        header = rows[header_idx]
        subprocess_name = header[0]
        n_cols = max(j for j, v in enumerate(header) if v is not None)
        columns = [str(v) for v in header[1:n_cols + 1]]
        # The sectors are the ranked columns ("Rank" row filled in), the
        # categories those that follow.
        n_sectors = sum(v is not None for v in rows[rank_idx][1:n_cols + 1])

        data_rows = rows[header_idx + 1:key_idx]
        values = np.array([[float(v) if v is not None else 0.0 for v in r[1:n_cols + 1]] for r in data_rows])
        labels = [r[0] for r in data_rows]
        sort_row = rows[0][1]
        if sort_row not in labels:
            raise ValueError(f"[{ws.title}] unknown sort row in B1: {sort_row!r} (expected: {labels})")
        sheets[subprocess_name] = {
            "sector_table": pd.DataFrame(values[:, :n_sectors], index=labels, columns=columns[:n_sectors]),
            "category_table": pd.DataFrame(values[:, n_sectors:], index=labels, columns=columns[n_sectors:]),
            "sort_row": sort_row,
            "sectors": columns[:n_sectors],
            "categories": columns[n_sectors:],
            # False for a workbook written before the categories were sorted by formulas
            "categories_sorted": any(label in first_col for label in category_rank_labels),
        }
    wb.close()
    return sheets


BUBBLE_ROWS = ["M_imp", "M_dom", "M_row", "Y_imp", "Y_dom"]


def select_bubble_items(sheet, type="categories", n=10):
    """Columns to plot for a sheet read back by load_subprocess_tables_excel.

    type="categories": all the consumption categories;
    type="products"  : the first n sectors.
    In both cases, columns in decreasing order of the sort row of the sheet
    (B1), ties in bridge order (same rule as the RANK + COUNTIF of the workbook).

    Returns:
        DataFrame rows BUBBLE_ROWS x selected columns.
    """
    if type == "categories":
        table, limit = sheet["category_table"], None
    elif type == "products":
        table, limit = sheet["sector_table"], n
    else:
        raise ValueError(f"type must be 'categories' or 'products', got {type!r}")
    order = np.argsort(-table.loc[sheet["sort_row"]].to_numpy(dtype=float), kind="stable")
    return table.iloc[:, order[:limit]].loc[BUBBLE_ROWS]


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
# FIGURES (from the Excel workbooks of export_subprocess_tables_excel)
# ============================================================================

PRODUCT_LABEL_MAX_CHARS = 30


def _short_label(label, max_chars=PRODUCT_LABEL_MAX_CHARS):
    """Truncate the EXIOBASE sector names (often very long) for the X axis."""
    label = str(label)
    return label if len(label) <= max_chars else label[:max_chars - 1] + "…"


def _scenario_label(excel_path):
    """Scenario name deduced from the file name (bubble_data_by_subprocess_<scenario>.xlsx)."""
    return Path(excel_path).stem.replace("bubble_data_by_subprocess_", "")


def _series_values(payload, components):
    """Ordinate of a series of BUBBLE_SERIES: sum of the `components` rows."""
    return payload.loc[list(components)].to_numpy(dtype=float).sum(axis=0)


def _shares(sheet, items, type):
    """Share (%) of each displayed item in the total footprint of the
    sub-process (TOTAL_FOOTPRINT_ROW, total = sum over all the sectors of the sheet)."""
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


def _subplot_title(subprocess_name, sheet):
    return f"{subprocess_name}\n(sorted by: {sheet['sort_row']})"


def _figure_subject(type, n):
    return "consumption categories" if type == "categories" else f"top {n} sectors per sub-process"


def plot_bubble_chart(base_excel_path, type="categories", n=10, scenarios=False,
                      comparison_excel_path=None, base_label=None, s1_label=None):
    """Bubble chart from the Excel workbooks of export_subprocess_tables_excel:
    1 subplot per sub-process, 2 bubbles per item of the X axis (see
    BUBBLE_SERIES): M_dom+M_imp (size Y_dom) and M_row (size Y_imp).

    Args:
        base_excel_path: workbook of the reference scenario (baseline).
        type: "categories" -> all the consumption categories;
            "products" -> the first `n` sectors of each sub-process.
            In both cases, in decreasing order of the sort row chosen in B1 of
            its sheet (editable by hand in Excel, see select_bubble_items).
        n: number of sectors displayed if type="products" (ignored otherwise).
        scenarios: False -> baseline alone; True -> comparison baseline vs
            `comparison_excel_path` (independent vertical axis per scenario).
            The displayed columns and their order are those of the baseline
            ranking, to compare the same items in both scenarios.
        base_label, s1_label: legends; by default deduced from the file names.

    Returns:
        (fig, axes)
    """
    base_sheets = load_subprocess_tables_excel(base_excel_path)
    base_label = base_label or _scenario_label(base_excel_path)
    base_payloads = {sp: select_bubble_items(sheet, type, n) for sp, sheet in base_sheets.items()}
    base_shares = {sp: _shares(base_sheets[sp], list(p.columns), type) for sp, p in base_payloads.items()}
    titles = {sp: _subplot_title(sp, sheet) for sp, sheet in base_sheets.items()}

    if not scenarios:
        return _plot_single(base_payloads, base_shares, titles, type, n, base_label)

    if comparison_excel_path is None:
        raise ValueError("scenarios=True requires comparison_excel_path")
    s1_sheets = load_subprocess_tables_excel(comparison_excel_path)
    s1_label = s1_label or _scenario_label(comparison_excel_path)
    s1_payloads, s1_shares = {}, {}
    for sp, base_payload in base_payloads.items():
        items = list(base_payload.columns)
        if sp in s1_sheets:
            table_key = "category_table" if type == "categories" else "sector_table"
            s1_payloads[sp] = s1_sheets[sp][table_key].reindex(index=BUBBLE_ROWS, columns=items).fillna(0.0)
            s1_shares[sp] = _shares(s1_sheets[sp], items, type)
        else:  # sub-process missing from the compared scenario: bubbles and bars drawn at 0
            s1_payloads[sp] = pd.DataFrame(0.0, index=BUBBLE_ROWS, columns=items)
            s1_shares[sp] = np.zeros(len(items))
    return _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                            base_label, s1_label)


def _plot_single(payloads, shares, titles, type, n, label):
    """1 subplot per sub-process, 1 scenario: share (%) bars in the total on
    the left axis, bubbles on the right axis. Bubble scale common to all the
    subplots (max of Y_dom, resp. Y_imp, over the displayed items)."""
    subprocesses = list(payloads)
    if not subprocesses:
        print("No data to plot.")
        return None, None

    n_items = max(p.shape[1] for p in payloads.values())
    fig, axes = _make_grid(len(subprocesses), n_items, type, base_width=6.2, base_height=4.8)
    refs = {row: _ref_max(payloads, row) for row in ("Y_dom", "Y_imp")}

    for i, sp in enumerate(subprocesses):
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

    for i in range(len(subprocesses), len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUBBLE_COLORS[key], markersize=10,
               alpha=0.7, label=f"{key} (size: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [Patch(facecolor=SHARE_BAR_COLOR, alpha=0.5, label=f"{SHARE_AXIS_LABEL} (left axis)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Bubble chart - {label} - {_figure_subject(type, n)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                     base_label, s1_label):
    """Comparison of two scenarios: share (%) bars in the total of each scenario
    on the left axis; bubbles on two independent right axes (base_label against
    the frame, s1_label shifted outwards), shifted horizontally. Bubble scale
    specific to each scenario, common to all the subplots."""
    subprocesses = list(base_payloads)
    if not subprocesses:
        print("No data to plot.")
        return None, None

    n_items = max(p.shape[1] for p in base_payloads.values())
    fig, axes = _make_grid(len(subprocesses), n_items, type, base_width=7.4, base_height=5.1)
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

    for i, sp in enumerate(subprocesses):
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

    for i in range(len(subprocesses), len(axes)):
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
    fig.suptitle(f"Bubble chart - {base_label} vs {s1_label} - {_figure_subject(type, n)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig, axes
