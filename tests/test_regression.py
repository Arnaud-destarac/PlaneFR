"""
Non-regression test of the totals behind the figures, on real data.

For the reference scenario of the default group (config.SCENARIO_GROUP, first
folder of get_scenario_folders) and for the 2019 geographies 2019_FR and 2019_W,
the footprint totals are computed with the current code and compared with the
snapshot tests/data/reference_totals.json:

- reference scenario: process_scenario (domestic, imported -- sums over the
  consumption categories -- and PBA), by indicator, as in the stacked bars
  and the Synthesis/Overshoot figures;
- 2019_FR, 2019_W: process_scenario_absolute (CBA, PBA) and
  process_scenario_per_capita (CBA, PBA), by indicator.

A difference means that a change of the code (or of the inputs: factors,
boundaries, engine results) changed the figures. Expected after a deliberate
change of data or method: regenerate the snapshot with

    PLANEFR_UPDATE_REFERENCE=1 py -m pytest tests/test_regression.py

(PowerShell: $env:PLANEFR_UPDATE_REFERENCE = "1"; py -m pytest tests/test_regression.py)

The test is skipped when the engine results (data/Outputs_MatMat, not
versioned) are missing, and when the snapshot is (re)written.
"""

import json
import math
import os
from pathlib import Path

import pytest

from planefr_lib import config, io, processing

SNAPSHOT_FILE = Path(__file__).resolve().parent / "data" / "reference_totals.json"
REGIONS_2019 = ["FR", "W"]
RELATIVE_TOLERANCE = 1e-9


def compute_totals():
    """{"<scenario>/<function>/<indicator>/<value>": float} for the reference
    scenario and the 2019 geographies REGIONS_2019."""
    try:
        reference = config.get_scenario_folders(exclude_2019=True)[config.REFERENCE_SCENARIO_IDX]
    except (ValueError, IndexError, FileNotFoundError) as error:
        pytest.skip(f"Engine results not available: {error}")
    folders_2019 = config.get_2019_folders(REGIONS_2019)
    if len(folders_2019) != len(REGIONS_2019):
        pytest.skip(f"2019 geographies {REGIONS_2019} not all available in {config.RAW_2019_DIR}")

    cf_df = io.load_characterization_factors()
    boundaries_df = io.load_boundaries()
    pop_df = io.load_population_df()
    totals = {}

    data, _ = processing.process_scenario(reference, cf_df, io.load_bridge_matrices(), boundaries_df)
    for indicator, payload in data.items():
        prefix = f"{reference.name}/process_scenario/{indicator}"
        totals[f"{prefix}/domestic"] = float(payload[processing.DOMESTIC].sum())
        totals[f"{prefix}/imported"] = float(payload[processing.IMPORTED].sum())
        totals[f"{prefix}/pba"] = float(payload["pba"])

    for folder in folders_2019:
        for name, data in [
            ("absolute", processing.process_scenario_absolute(folder, cf_df, boundaries_df)),
            ("per_capita", processing.process_scenario_per_capita(folder, cf_df, boundaries_df, pop_df)),
        ]:
            for indicator, payload in data.items():
                prefix = f"{folder.name}/{name}/{indicator}"
                totals[f"{prefix}/cba"] = processing.extract_value(payload, "cba")
                totals[f"{prefix}/pba"] = processing.extract_value(payload, "pba")
    return totals


def write_snapshot(totals):
    SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_FILE.write_text(json.dumps(dict(sorted(totals.items())), indent=1, ensure_ascii=False),
                             encoding="utf-8")


def same(a, b):
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=RELATIVE_TOLERANCE, abs_tol=1e-300)


def test_footprint_totals_match_snapshot():
    totals = compute_totals()
    if os.environ.get("PLANEFR_UPDATE_REFERENCE") == "1" or not SNAPSHOT_FILE.exists():
        write_snapshot(totals)
        pytest.skip(f"Snapshot written ({len(totals)} values): {SNAPSHOT_FILE}")

    expected = json.loads(SNAPSHOT_FILE.read_text(encoding="utf-8"))
    missing = sorted(set(expected) - set(totals))
    added = sorted(set(totals) - set(expected))
    changed = [f"{key}: {expected[key]!r} -> {totals[key]!r}"
               for key in sorted(set(expected) & set(totals)) if not same(expected[key], totals[key])]
    assert not (missing or added or changed), "\n".join(
        [f"{len(changed)} changed value(s):", *changed[:30],
         f"missing: {missing[:30]}", f"added: {added[:30]}",
         "Deliberate change? Regenerate with PLANEFR_UPDATE_REFERENCE=1 (see module docstring)."])
