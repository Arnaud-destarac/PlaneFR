"""
Tests of the Excel reads of planefr_lib.io that locate their table by content
(header row, product column) rather than by position: budget_shares.xlsx
(budget shares, population) and the M / M_split sheets of bridge_matrices.xlsx.

Small workbooks are written in a temporary folder and config points to them.
"""

import pandas as pd
import pytest

from planefr_lib import config, io


def write_sheets(path, sheets):
    """sheets: {name: list of rows} written without header/index."""
    with pd.ExcelWriter(path) as writer:
        for name, rows in sheets.items():
            pd.DataFrame(rows).to_excel(writer, sheet_name=name, header=False, index=False)


def shares_rows(title_rows):
    return [*title_rows,
            ["EXIOBASE code", "Region", "EPC share ref. (SSP2, 2019)", "CTR share ref."],
            ["FR", "France", 0.01, 0.02],
            ["W", "Total 49 regions", 1.0, 1.0],
            ["TW", "Taiwan", "n.a.", 0.005],
            ["Note: shares computed from ...", None, None, None]]


def population_rows(title_rows):
    return [*title_rows,
            ["EXIOBASE code", "Region", "UN – Population 2019 (thousands)", "UN – Population 2050 (thousands)"],
            ["FR", "France", 65000.0, 66000.0],
            [" W ", "World", 7.7e6, 9.7e6]]


@pytest.mark.parametrize("title_rows", [
    [["Title"], ["Note"], [None]],      # current layout: header on the 4th row
    [["Title"]],                        # a note row removed
    [["Title"], ["Note"], ["Note 2"], [None], [None]],  # rows added
])
def test_budget_shares_header_found_by_content(tmp_path, monkeypatch, title_rows):
    path = tmp_path / "budget_shares.xlsx"
    write_sheets(path, {config.BUDGET_SHARES_SHEET: shares_rows(title_rows),
                        config.POPULATION_SHEET: population_rows(title_rows)})
    monkeypatch.setattr(config, "BUDGET_SHARES_FILE", path)

    shares = io.load_budget_shares_df()
    assert list(shares.index) == ["FR", "W", "TW"]  # note row dropped
    assert shares.loc["FR", "EPC share ref. (SSP2, 2019)"] == 0.01
    assert pd.isna(shares.loc["TW", "EPC share ref. (SSP2, 2019)"])  # "n.a."

    pop = io.load_population_df()
    assert list(pop.columns) == ["2019", "2050"]
    assert pop.loc["W", "2050"] == 9.7e6  # code stripped


def test_budget_shares_missing_header_raises(tmp_path, monkeypatch):
    path = tmp_path / "budget_shares.xlsx"
    rows = shares_rows([["Title"]])
    rows[1][0] = "Code"  # header renamed
    write_sheets(path, {config.BUDGET_SHARES_SHEET: rows})
    monkeypatch.setattr(config, "BUDGET_SHARES_FILE", path)
    with pytest.raises(ValueError, match="Header row"):
        io.load_budget_shares_df()


def test_budget_shares_duplicated_code_raises(tmp_path, monkeypatch):
    path = tmp_path / "budget_shares.xlsx"
    rows = shares_rows([["Title"]]) + [["FR", "France again", 0.5, 0.5]]
    write_sheets(path, {config.BUDGET_SHARES_SHEET: rows})
    monkeypatch.setattr(config, "BUDGET_SHARES_FILE", path)
    with pytest.raises(ValueError, match="Duplicated region codes"):
        io.load_budget_shares_df()


def bridge_m_rows(n_products, extra_columns=0):
    """Layout of the M sheets: header (…, sector, categories), 2nd header row
    (category, sub_category, sector), then one row per product."""
    pad = [None] * extra_columns
    rows = [[*pad, None, None, "sector", "Food", "Housing", None],
            [*pad, "category", "sub_category", "sector", None, None, None]]
    rows += [[*pad, "Agri", None, f"product {i}", i % 2, 1 - i % 2, None] for i in range(n_products)]
    return rows


@pytest.mark.parametrize("extra_columns, title_rows", [(0, 0), (2, 0), (0, 2)])
def test_bridge_m_matrix_by_content(tmp_path, monkeypatch, extra_columns, title_rows):
    path = tmp_path / "bridge_matrices.xlsx"
    rows = [["Title"]] * title_rows + bridge_m_rows(5, extra_columns)
    write_sheets(path, {"M": rows})
    monkeypatch.setattr(config, "BRIDGE_MATRICES_FILE", path)
    monkeypatch.setattr(config, "N_PRODUCTS", 5)

    m = io.load_bridge_m_matrix("M")
    assert list(m.columns) == ["Food", "Housing"]
    assert list(m.index) == [f"product {i}" for i in range(5)]
    assert m.loc["product 1"].tolist() == [1, 0]


@pytest.mark.parametrize("n_products", [4, 6])
def test_bridge_m_matrix_wrong_product_count_raises(tmp_path, monkeypatch, n_products):
    path = tmp_path / "bridge_matrices.xlsx"
    write_sheets(path, {"M": bridge_m_rows(n_products)})
    monkeypatch.setattr(config, "BRIDGE_MATRICES_FILE", path)
    monkeypatch.setattr(config, "N_PRODUCTS", 5)
    with pytest.raises(ValueError, match="expected 5 distinct products"):
        io.load_bridge_m_matrix("M")
