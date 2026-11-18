"""
Excel export of the data shown in a figure, for publishing the results.

Each figure function (create_<notebook>...) takes an `excel_path` argument:
when given, it writes there a workbook with the values it plots, through
write_figure_excel:
  - "About" sheet: what the figure shows, units, parameters (sharing
    principle, boundaries...);
  - one sheet per table, in "long" format (one row per value, explicit
    columns), easy to filter or to reload with pandas.
The notebooks save it next to the figure, with the same name (.xlsx).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import colors, config, processing
from .processing import DOMESTIC, IMPORTED

# Display label of the consumption categories (same as the legends of the
# figures): F_Y = direct pressures of final demand, not broken down by the bridge.
CATEGORY_LABELS = {"F_Y": "Not allocated"}

ORIGIN_LABELS = {DOMESTIC: "Domestic", IMPORTED: "Imported"}

# Columns wider than this are not widened further (long sector names).
MAX_COLUMN_WIDTH = 60


def category_label(category):
    """Label of a consumption category in the figures and the workbooks."""
    return CATEGORY_LABELS.get(category, category)


def clean_value(value):
    """float, or None for a missing/non-finite value (empty cell in Excel)."""
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def variation_pct(value, reference):
    """Variation (%) of value relative to reference; None if not computable."""
    value, reference = clean_value(value), clean_value(reference)
    if value is None or reference is None or reference == 0:
        return None
    return (value - reference) / reference * 100


def ratio(value, reference):
    """value / reference; None if not computable."""
    value, reference = clean_value(value), clean_value(reference)
    if value is None or reference is None or reference == 0:
        return None
    return value / reference


def indicator_unit(boundaries_df, indicator_name, unit_row=config.UNIT_ROW_ABS):
    """Display unit of an indicator ("Figures unit" row by default)."""
    return processing.lookup_first_text(boundaries_df, unit_row, indicator_name) or ""


def footprint_rows(scenario_name, data_by_indicator, boundaries_df, indicators=None):
    """Rows (dicts) of the consumption-based footprint of a scenario, broken
    down by indicator, consumption category and origin (domestic/imported),
    in the unit of the figures -- the segments of the stacked bars.

    Args:
        data_by_indicator: {indicator_name: payload of processing.process_scenario}
        indicators: indicators to export (default: all, in their order).
    """
    rows = []
    for indicator_name in indicators or list(data_by_indicator):
        payload = data_by_indicator.get(indicator_name)
        if payload is None:
            continue
        unit = indicator_unit(boundaries_df, indicator_name)
        total = processing.extract_value(payload, "cba")
        for category in colors.sort_categories(list(payload["categories"])):
            for origin in (DOMESTIC, IMPORTED):
                value = clean_value(payload[origin].get(category, 0.0))
                share = ratio(value, total)
                rows.append({
                    "Scenario": scenario_name,
                    "Indicator": indicator_name,
                    "Unit": unit,
                    "Consumption category": category_label(category),
                    "Origin": ORIGIN_LABELS[origin],
                    "Value": value,
                    "Share of the total footprint (%)": 100 * share if share is not None else None,
                })
    return rows


def write_figure_excel(excel_path, sheets, about=None):
    """Write the data of a figure into an Excel workbook.

    Args:
        excel_path: output path (.xlsx); the folder is created if needed.
        sheets: {sheet name: DataFrame or list of row dicts}, written in this
            order, without index. Names are truncated to 31 characters (Excel).
        about: optional {label: value} written first, in an "About" sheet.

    Returns:
        Path of the written workbook.
    """
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        if about:
            about_df = pd.DataFrame([(k, "" if v is None else v) for k, v in about.items()],
                                    columns=["Item", "Value"])
            about_df.to_excel(writer, sheet_name="About", index=False)
        for name, table in sheets.items():
            df = table if isinstance(table, pd.DataFrame) else pd.DataFrame(table)
            df.to_excel(writer, sheet_name=str(name)[:31], index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            for column_cells in ws.columns:
                width = max((len(str(c.value)) for c in column_cells if c.value is not None), default=8)
                ws.column_dimensions[column_cells[0].column_letter].width = min(width + 2, MAX_COLUMN_WIDTH)
    return excel_path
