"""
Unit tests of planefr_lib.processing on small synthetic tables: filtering and
weighting by characterization factor, and the footprint functions
(process_single_indicator_scenario, process_scenario, process_scenario_absolute,
process_scenario_per_capita, process_indicator_lp_breakdown) on scenario
folders written in a temporary directory.

No dependency on the real data of data/.
"""

import numpy as np
import pandas as pd
import pytest

from planefr_lib import config, processing

IND = config.CF_INDICATOR_COL
LP = config.CF_LP_COL
STRESSOR = config.CF_STRESSOR_COL
FACTOR = config.CF_FACTOR_COL


def reference_filter_and_weight(df, extension_factor_map):
    """Row-by-row implementation (former version of filter_and_weight),
    kept as the reference of the vectorized one."""
    rows = [row * extension_factor_map[idx[0] if isinstance(idx, tuple) else idx]
            for idx, row in df.iterrows()
            if (idx[0] if isinstance(idx, tuple) else idx) in extension_factor_map]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# === filter_and_weight ===

def test_filter_and_weight_simple_index():
    df = pd.DataFrame({"s1": [1.0, 2.0, 3.0], "s2": [10.0, 20.0, 30.0]}, index=["CO2", "CH4", "N2O"])
    out = processing.filter_and_weight(df, {"CO2": 1.0, "CH4": 28.0, "SF6": 1e4})
    assert list(out.index) == ["CO2", "CH4"]
    assert out.loc["CH4"].tolist() == [56.0, 560.0]
    assert out.loc["CO2"].tolist() == [1.0, 10.0]


def test_filter_and_weight_multiindex_uses_first_level():
    index = pd.MultiIndex.from_tuples([("CO2", "air"), ("CH4", "air"), ("CH4", "soil")])
    df = pd.DataFrame({"s1": [1.0, 1.0, 2.0]}, index=index)
    out = processing.filter_and_weight(df, {"CH4": 10.0})
    assert out["s1"].tolist() == [10.0, 20.0]
    assert out.index.equals(index[1:])


def test_filter_and_weight_no_match_or_empty():
    df = pd.DataFrame({"s1": [1.0]}, index=["CO2"])
    assert processing.filter_and_weight(df, {"CH4": 1.0}).empty
    assert processing.filter_and_weight(df, {}).empty
    assert processing.filter_and_weight(df.iloc[:0], {"CO2": 1.0}).empty


def test_filter_and_weight_duplicated_stressors():
    df = pd.DataFrame({"s1": [1.0, 2.0]}, index=["CO2", "CO2"])
    assert processing.filter_and_weight(df, {"CO2": 3.0})["s1"].tolist() == [3.0, 6.0]


@pytest.mark.parametrize("multiindex", [False, True])
def test_filter_and_weight_matches_row_by_row_version(multiindex):
    rng = np.random.default_rng(0)
    stressors = [f"st{i}" for i in range(50)]
    columns = pd.MultiIndex.from_product([["FR", "DE"], [f"sector{j}" for j in range(7)]],
                                         names=["region", "sector"])
    index = (pd.MultiIndex.from_arrays([stressors, ["unit"] * 50]) if multiindex else pd.Index(stressors))
    df = pd.DataFrame(rng.normal(size=(50, len(columns))), index=index, columns=columns)
    factors = {s: float(rng.uniform(-2, 5)) for s in rng.choice(stressors, 20, replace=False)}
    factors["absent"] = 1.0

    expected = reference_filter_and_weight(df, factors)
    out = processing.filter_and_weight(df, factors)
    np.testing.assert_allclose(out.to_numpy(), expected.to_numpy(), rtol=0, atol=0)
    assert list(out.columns) == list(expected.columns)
    np.testing.assert_allclose(processing.filter_and_weight_vector(df, factors).to_numpy(),
                               reference_filter_and_weight(df, factors).sum().to_numpy())
    assert processing.filter_and_weight_scalar(df, factors) == pytest.approx(expected.to_numpy().sum())


def test_filter_and_weight_vector_keeps_last_column_level():
    columns = pd.MultiIndex.from_tuples([("FR", "Food"), ("FR", "Housing")])
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"], columns=columns)
    out = processing.filter_and_weight_vector(df, {"a": 1.0, "b": 10.0})
    assert out.to_dict() == {"Food": 31.0, "Housing": 42.0}
    assert processing.filter_and_weight_vector(df, {"z": 1.0}).empty


def test_filter_and_weight_scalar():
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"])
    assert processing.filter_and_weight_scalar(df, {"b": 2.0}) == 14.0
    assert processing.filter_and_weight_scalar(df, {"z": 2.0}) == 0.0


# === Characterization factors ===

@pytest.fixture
def cf_df():
    return pd.DataFrame({
        IND: ["Climate", "Climate", "Climate", "Water"],
        LP: ["ghg", "ghg", "land", "water"],
        STRESSOR: ["CO2", "CH4", "CO2", "Blue"],
        FACTOR: [1.0, 28.0, 2.0, 1.0],
    })


def test_get_unique_indicators(cf_df):
    assert processing.get_unique_indicators(cf_df) == {"Climate": ["ghg", "land"], "Water": ["water"]}


def test_build_extension_factor_map(cf_df):
    # Without LP, the last row of a stressor wins (CO2 of "land")
    assert processing.build_extension_factor_map(cf_df, "Climate") == {"CO2": 2.0, "CH4": 28.0}
    assert processing.build_extension_factor_map(cf_df, "Climate", "ghg") == {"CO2": 1.0, "CH4": 28.0}
    assert processing.build_extension_factor_map(cf_df, "Unknown") == {}


# === Footprints on synthetic scenario folders ===

SECTORS = ["s1", "s2"]
CATEGORIES = ["Food", "Housing"]


def write_pickle(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)


def stressor_table(values, stressors=("CO2", "CH4"), columns=SECTORS):
    return pd.DataFrame(values, index=list(stressors), columns=columns, dtype=float)


@pytest.fixture
def boundaries_df():
    return pd.DataFrame({"Climate": [10.0, 1000.0], "Water": [np.nan, np.nan]},
                        index=[config.CONVERSION_ROW_ABS, config.CONVERSION_ROW_PER_CAPITA])


@pytest.fixture
def bridge_df():
    """Same layout as bridge_matrices.xlsx read by io.load_bridge_matrices:
    categories as column labels from column 6, 1st row ignored, sectors in
    column 5 (s1 -> Food, s2 -> half/half)."""
    rows = [[None] * 8,
            [None] * 5 + ["s1", 1.0, 0.0],
            [None] * 5 + ["s2", 0.5, 0.5]]
    return pd.DataFrame(rows, columns=[f"c{i}" for i in range(6)] + CATEGORIES)


@pytest.fixture
def france_scenario(tmp_path):
    """France scenario 2050: LP ghg (dom/imp split) + F_Y_tot, LP land without F_Y."""
    scen = tmp_path / "S1"
    ext = scen / "extensions"
    for lp, dom, imp, f_x in [
        ("ghg", [[1, 2], [1, 0]], [[3, 4], [0, 1]], [[5, 5], [1, 1]]),
        ("land", [[1, 1], [0, 0]], [[0, 0], [0, 0]], [[1, 0], [0, 0]]),
    ]:
        write_pickle(ext / f"dom_{lp}" / "d_cba_k.pkl", stressor_table(dom))
        write_pickle(ext / f"imp_{lp}" / "d_cba_k.pkl", stressor_table(imp))
        write_pickle(ext / f"dom_{lp}" / "d_cba.pkl", stressor_table(dom))
        write_pickle(ext / f"imp_{lp}" / "d_cba.pkl", stressor_table(imp))
        write_pickle(ext / f"dom_{lp}" / "F_x_dom.pkl", stressor_table(f_x))
    write_pickle(ext / "dom_ghg" / "F_Y_tot.pkl", stressor_table([[7], [1]], columns=["total"]))
    return scen


def test_process_single_indicator_scenario(france_scenario, cf_df, bridge_df, boundaries_df):
    out = processing.process_single_indicator_scenario(
        "Climate", "ghg", france_scenario, cf_df, bridge_df, boundaries_df)
    # Factors without LP restriction: CO2 -> 2 (land row), CH4 -> 28; conversion 10
    # dom by sector: s1 = 1*2 + 1*28 = 30, s2 = 2*2 = 4 -> Food 32, Housing 2
    assert out[processing.DOMESTIC].to_dict() == pytest.approx({"Food": 3.2, "Housing": 0.2, "F_Y": 4.2})
    # imp: s1 = 6, s2 = 8 + 28 = 36 -> Food 24, Housing 18
    assert out[processing.IMPORTED].to_dict() == pytest.approx({"Food": 2.4, "Housing": 1.8, "F_Y": 0.0})
    # PBA: F_x_dom (10 + 10 + 28 + 28) + F_Y (14 + 28), / 10
    assert out["pba"] == pytest.approx(11.8)


def test_process_single_indicator_scenario_f_y_already_in_d_cba(france_scenario, cf_df, bridge_df,
                                                                 boundaries_df):
    (france_scenario / "extensions" / "dom_ghg" / "S_Y.pkl").write_bytes(b"")
    out = processing.process_single_indicator_scenario(
        "Climate", "ghg", france_scenario, cf_df, bridge_df, boundaries_df)
    assert "F_Y" not in out[processing.DOMESTIC].index
    assert out["pba"] == pytest.approx(11.8)  # F_Y still counted in the PBA


def test_process_scenario_aggregates_lps(france_scenario, cf_df, bridge_df, boundaries_df):
    data, indicator_to_lp = processing.process_scenario(france_scenario, cf_df, bridge_df, boundaries_df)
    assert indicator_to_lp == {"Climate": ["ghg", "land"], "Water": ["water"]}
    assert set(data) == {"Climate"}  # no water files
    climate = data["Climate"]
    # land: dom s1 = 2, s2 = 2 -> Food 3, Housing 1 (/10)
    assert climate[processing.DOMESTIC]["Food"] == pytest.approx(3.2 + 0.3)
    assert climate[processing.DOMESTIC]["Housing"] == pytest.approx(0.2 + 0.1)
    assert climate["pba"] == pytest.approx(11.8 + 0.2)
    assert set(climate["categories"]) == {"Food", "Housing", "F_Y"}


def test_process_scenario_absolute_france(france_scenario, cf_df, boundaries_df):
    data = processing.process_scenario_absolute(france_scenario, cf_df, boundaries_df)
    # Factors restricted to each LP: ghg CO2 -> 1, CH4 -> 28; land CO2 -> 2
    # CBA ghg: dom (3 + 28) + imp (7 + 28) + F_Y (7 + 28) = 101; land: 2 * 2 = 4
    # PBA ghg: F_x_dom (10 + 56) + F_Y 35 = 101; land: 2
    climate = data["Climate"]
    assert processing.extract_value(climate, "cba") == pytest.approx(10.5)
    assert processing.extract_value(climate, "pba") == pytest.approx(10.3)
    assert climate["categories"] == ["total"]


def test_process_scenario_per_capita(france_scenario, cf_df, boundaries_df):
    pop_df = pd.DataFrame({"2050": [2.0]}, index=["FR"])  # thousands -> 2000 persons
    data = processing.process_scenario_per_capita(france_scenario, cf_df, boundaries_df, pop_df)
    assert processing.extract_value(data["Climate"], "cba") == pytest.approx(105 / 1000 / 2000)
    assert data["Climate"]["pba"] == pytest.approx(103 / 1000 / 2000)


def test_process_indicator_lp_breakdown(france_scenario, cf_df, boundaries_df):
    out = processing.process_indicator_lp_breakdown("Climate", france_scenario, cf_df, boundaries_df)
    assert out == pytest.approx({"ghg": 10.1, "land": 0.4})


def test_world_scenario_without_dom_imp_split(tmp_path, cf_df, boundaries_df):
    scen = tmp_path / "2019_W"
    ext = scen / "extensions"
    for lp in ("ghg", "ghg_combustion"):
        write_pickle(ext / lp / "d_cba.pkl", stressor_table([[1, 1], [1, 0]]))
        write_pickle(ext / lp / "F_x_dom.pkl", stressor_table([[2, 0], [0, 0]]))
    write_pickle(ext / "ghg" / "F_Y_tot.pkl", stressor_table([[1], [0]], columns=["total"]))
    cf = pd.concat([cf_df, pd.DataFrame({IND: ["Climate"], LP: ["ghg_combustion"],
                                         STRESSOR: ["CO2"], FACTOR: [1.0]})])

    assert not processing.scenario_has_dom_imp_split(scen)
    data = processing.process_scenario_absolute(scen, cf, boundaries_df)
    # ghg only (ghg_combustion ignored for World/Europe): CBA 2 + 28 + 1, PBA 2 + 1
    assert processing.extract_value(data["Climate"], "cba") == pytest.approx(3.1)
    assert processing.extract_value(data["Climate"], "pba") == pytest.approx(0.3)


def test_missing_d_cba_k_gives_no_result(tmp_path, cf_df, bridge_df, boundaries_df):
    assert processing.process_single_indicator_scenario(
        "Climate", "ghg", tmp_path, cf_df, bridge_df, boundaries_df) is None
