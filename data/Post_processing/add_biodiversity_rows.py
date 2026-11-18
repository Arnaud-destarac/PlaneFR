"""
Adds to the "All" sheet of indicator_computing.xlsx one row per column
(exiobase stressor) of "impact_world_plus_2.2.1_expert_version_exiobase_3.9_and_after.xlsx"
whose sum over the rows in PDF.m2.yr units (biodiversity metric of Impact World+)
is non-zero.

Only the rows whose label ends with "(PDF.m2.yr)" are summed: these are the
biodiversity indicators (the other rows of the file - DALY, CTUh, kg eq, m3 eq,
etc. - relate to other impact categories and must not be mixed in).

For each retained column:
    - "Indicators"                -> BIODIVERSITY_INDICATOR
    - "EXIOBASE groups"           -> left empty (to be filled in by hand: LP of the stressor)
    - "EXIOBASE stressors"        -> name of the column
    - "Units"                     -> left empty
    - "Characterization factors"  -> sum of the values (PDF.m2.yr rows) of the column

The Impact World+ file is read in the input_data/ folder next to this script.
"""

import shutil
from pathlib import Path

import openpyxl

DATA_DIR = Path(__file__).resolve().parent
CF_PATH = DATA_DIR / "indicator_computing.xlsx"
IMPACT_WORLD_PATH = DATA_DIR / "input_data" / "impact_world_plus_2.2.1_expert_version_exiobase_3.9_and_after.xlsx"

SHEET_CF = "All"
SHEET_IMPACT_WORLD = "Sheet1"

# Columns of the "All" sheet of indicator_computing.xlsx
COL_INDICATOR = 1  # A
COL_EXIOBASE_GROUP = 2  # B
COL_EXIOBASE_STRESSOR = 3  # C
COL_UNIT = 4  # D
COL_CHARACTERIZATION_FACTOR = 5  # E

BIODIVERSITY_INDICATOR = "Biodiversity loss - PDF"


PDF_UNIT_SUFFIX = "(PDF.m2.yr)"
EXCLUDED_SUBSTRING = "long term"


def get_biodiversity_rows(ws):
    """Return the row indices whose label is a PDF.m2.yr (biodiversity) indicator,
    excluding any indicator whose name contains "long term"."""
    rows = []
    for r in range(2, ws.max_row + 1):
        label = ws.cell(row=r, column=1).value
        if not isinstance(label, str):
            continue
        if not label.strip().endswith(PDF_UNIT_SUFFIX):
            continue
        if EXCLUDED_SUBSTRING in label.lower():
            continue
        rows.append(r)
    return rows


def compute_nonzero_column_sums(ws, biodiversity_rows):
    """Return list of (column_name, sum) for each data column whose sum over the
    biodiversity (PDF.m2.yr) rows is non-zero."""
    max_col = ws.max_column
    results = []
    for c in range(2, max_col + 1):  # column 1 is the row-label column
        colname = ws.cell(row=1, column=c).value
        total = 0.0
        for r in biodiversity_rows:
            v = ws.cell(row=r, column=c).value
            if isinstance(v, (int, float)):
                total += v
        if total != 0:
            results.append((colname, total))
    return results


def main():
    backup_path = CF_PATH.with_name(CF_PATH.stem + " (backup before biodiversity loss rows).xlsx")
    if backup_path.exists():
        # Always start again from the pristine backup to avoid stacking rows
        # at each new run of the script (e.g. after fixing the computation).
        shutil.copy2(backup_path, CF_PATH)
        print(f"File restored from the backup: {backup_path.name}")
    else:
        shutil.copy2(CF_PATH, backup_path)
        print(f"Backup created: {backup_path.name}")

    wb_impact = openpyxl.load_workbook(IMPACT_WORLD_PATH, data_only=True)
    ws_impact = wb_impact[SHEET_IMPACT_WORLD]
    biodiversity_rows = get_biodiversity_rows(ws_impact)
    print(f"{len(biodiversity_rows)} '(PDF.m2.yr)' rows identified as biodiversity indicators.")
    nonzero_sums = compute_nonzero_column_sums(ws_impact, biodiversity_rows)
    print(f"{len(nonzero_sums)} columns with a non-zero sum out of {ws_impact.max_column - 1} columns.")

    wb_cf = openpyxl.load_workbook(CF_PATH, data_only=False)
    ws_cf = wb_cf[SHEET_CF]

    next_row = ws_cf.max_row + 1
    for offset, (colname, total) in enumerate(nonzero_sums):
        r = next_row + offset
        ws_cf.cell(row=r, column=COL_INDICATOR, value=BIODIVERSITY_INDICATOR)
        ws_cf.cell(row=r, column=COL_EXIOBASE_GROUP, value=None)
        ws_cf.cell(row=r, column=COL_EXIOBASE_STRESSOR, value=colname)
        ws_cf.cell(row=r, column=COL_UNIT, value=None)
        ws_cf.cell(row=r, column=COL_CHARACTERIZATION_FACTOR, value=total)

    wb_cf.save(CF_PATH)
    print(f"{len(nonzero_sums)} rows added to sheet '{SHEET_CF}' (rows {next_row} to {next_row + len(nonzero_sums) - 1}).")


if __name__ == "__main__":
    main()
