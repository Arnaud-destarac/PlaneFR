#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlaneFR – Planetary budget shares by EXIOBASE 3 region
=======================================================

Computes the allocation shares of the planetary budget under two equity
principles, for the 49 EXIOBASE 3 regions and the EU27 aggregate, under the
five SSP scenarios and for all the reference periods considered.

    EPC_s(P) = Pop_s(P) / Σ_s Pop_s(P)
    CTR_s(P, G) = [Pop_s(P) / GDPcap_s(G)] / Σ_s [Pop_s(P) / GDPcap_s(G)]

    with GDPcap_s(G) = Σ_t GDP_t / Σ_t Pop_t over period G

The second expression is the reduced form of
CTR_s = (GDPcap_W / GDPcap_s) × (Pop_s / Pop_W) × α_AtP,
where α_AtP = Pop_W / (GDPcap_W × Σ_s Pop_s / GDPcap_s): the world terms
cancel out, the two expressions are strictly equivalent.

Regional shares are the sum of the shares computed at country level, the only
aggregation consistent with the principle (the Pop/GDPcap indicator is not
additive).

Periods
-------
    Demographic (P) : 2019 · 2019-2050 · 2019-2100
    GDP (G)         : 1990-2019 · 1990-2050 · 2019 · 2019-2050 · 2019-2100 · 1990-2100

That is 3 EPC variants and 18 CTR variants, for each of the 5 SSPs.
Reference setting: SSP2, EPC 2019 and CTR Pop 2019 × GDP 1990-2019 -- it only
uses observed data and therefore does not depend on any scenario.

Sources
-------
    History 1990-2025: World Bank, World Development Indicators
        GDP         NY.GDP.MKTP.PP.KD  (PPP, constant 2021 international dollars)
        Population  SP.POP.TOTL
    Projections 2026-2100: IIASA SSP database, 2025 update
        GDP         OECD ENV-Growth model, variable GDP|PPP (USD 2017 PPP)
        Population  IIASA-Wittgenstein Centre (IIASA-WiC POP 2025)
    UN population: World Population Prospects 2024 (estimates + medium variant)

Splicing of the projections
---------------------------
For each economy, each variable and each scenario k:

    X_t = X_WB(t0) × S_k(t) / S_k(t0)     for t > t0

where t0 is the last year observed by the World Bank and S_k the SSP
trajectory, geometrically interpolated between five-year nodes. The level
remains the World Bank one; only the dynamics are borrowed from the SSPs, with
no PPP base conversion -- the ratio S_k(t)/S_k(t0) is invariant to it.

GDP / population consistency: in each scenario, GDPcap_s is computed with the
population of the same SSP as the GDP, since the SSP GDP was projected with the
corresponding SSP demography.

Implementation notes
--------------------
  - France: the SSP dynamics are computed on the sum France + French Guiana +
    Mayotte, separate entities in the IIASA database, to approach the World
    Bank perimeter, which includes the overseas departments. Guadeloupe,
    Martinique and La Réunion have no entity of their own at IIASA and are not
    distinguished.
  - Regional fallback: economies without their own SSP projection follow the
    aggregate SSP dynamics of their EXIOBASE region, computed over the
    economies of the region that have their own projection.
  - Perimeter: individual economies are separated from the WDI aggregates by
    membership in the ISO 3166-1 standard, not by a fixed list of aggregates;
    the result is checked by a demographic closure test on the "World"
    aggregate.
  - Excluded economies: those without World Bank GDP over 1990-2025 are left
    out of all calculations, EPC included -- EPC and CTR cover the same set.
  - Economies with partial coverage have GDPcap_s computed on the available
    years of each period only (population sums restricted to the years with
    GDP).

Usage
-----
    python budget_shares.py [--outdir DIR] [--check WORKBOOK.xlsx]

The script is self-contained: the five source extractions are enough to
reproduce all the results, without any intermediate file.

Dependencies: openpyxl, country_converter (pip install openpyxl country_converter)
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

logging.disable(logging.WARNING)  # country_converter is verbose about unmatched names

import country_converter as coco
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────── Configuration ────────────────────────────────

DATA = Path(__file__).resolve().parent
SRC = {
    "gdp_wb": DATA / "Budget share" / "GDP PPP BM.csv",
    "pop_wb": DATA / "Budget share" / "Population BM.csv",
    "gdp_ssp": DATA / "Budget share" / "GDP PPP SSPs.csv",
    "pop_ssp": DATA / "Budget share" / "Population SSPs.csv",
    "pop_un": DATA / "Budget share" / "Population ONU.xlsx"
}

# Sheet names. SHEET_VARIANTS and SHEET_POPULATION are referenced by cell in
# thresholds.xlsx (external links "[1]Budget_shares_variants!$E$15",
# "[1]Population!$C$15"...): do not rename them nor move their columns/rows
# without updating those formulas.
SHEET_REGIONS = "Budget_shares_regions"
SHEET_VARIANTS = "Budget_shares_variants"
SHEET_POPULATION = "Population"
SHEET_DIAGNOSTICS = "Country_diagnostics"


# Economies of the WDI extractions whose code does not belong to the ISO 3166-1
# standard and that the perimeter filter (see derive_perimeter) would therefore
# wrongly treat as aggregates. CHI groups Guernsey and Jersey, economies not
# distinguished by the WDI.
NON_ISO_ECONOMIES = {"CHI"}

# Tolerance of the demographic closure check: allowed gap between the sum of the
# retained populations and the WDI "World" aggregate (economies with missing
# population, rounding). A regional aggregate kept by mistake would blow up this gap.
PERIMETER_TOLERANCE = 0.02

SSPS = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
NODES = list(range(2020, 2101, 5))          # SSP nodes: 2020 (Historical Reference), then scenario
HIST_FIRST, HIST_LAST = 1990, 2025          # World Bank coverage
LAST_YEAR = 2100

PP = [("2019", 2019, 2019), ("2019-2050", 2019, 2050), ("2019-2100", 2019, 2100)]
# Demographic periods of the EPC shares based on the UN population
UN_PP = [("2019", 2019, 2019), ("2050", 2050, 2050), ("2019-2100", 2019, 2100)]
# Periods of the "Population" sheet (UN population, mean over the period): the
# three UN_PP periods first (columns C-E), then those used by planefr_lib
# (base year 2015 and the "Equality 2050" mean).
POP_PP = UN_PP + [("2015", 2015, 2015), ("2019-2050", 2019, 2050)]
GP = [("1990-2019", 1990, 2019), ("1990-2050", 1990, 2050), ("2019", 2019, 2019),
      ("2019-2050", 2019, 2050), ("2019-2100", 2019, 2100), ("1990-2100", 1990, 2100)]
COMBOS = [(p, g) for p in range(len(PP)) for g in range(len(GP))]

REF_SSP = "SSP2"
REF_P, REF_G = 0, 0                         # Pop 2019 × GDP 1990-2019
REF_COMBO = REF_P * len(GP) + REF_G

UNIT_GDP_SSP = "billion USD_2017/yr"
UNIT_POP_SSP = "million"

FR_PARTS = ["France", "French Guiana", "Mayotte"]
DROM = ["GLP", "MTQ", "GUF", "REU", "MYT"]
ROW_REGIONS = ["WA", "WE", "WF", "WL", "WM"]  # "Rest of World" regions, used for the fallback

# Status of the dynamics applied to an economy (Country_diagnostics sheet)
STATUS_OWN = "Own SSP"
STATUS_FALLBACK = "EXIOBASE region fallback"
STATUS_NONE = "None"

REGION_LABELS = [
    ("AT", "Austria"), ("BE", "Belgium"), ("BG", "Bulgaria"), ("CY", "Cyprus"),
    ("CZ", "Czechia"), ("DE", "Germany"), ("DK", "Denmark"), ("EE", "Estonia"),
    ("ES", "Spain"), ("FI", "Finland"), ("FR", "France"), ("GR", "Greece"),
    ("HR", "Croatia"), ("HU", "Hungary"), ("IE", "Ireland"), ("IT", "Italy"),
    ("LT", "Lithuania"), ("LU", "Luxembourg"), ("LV", "Latvia"), ("MT", "Malta"),
    ("NL", "Netherlands"), ("PL", "Poland"), ("PT", "Portugal"), ("RO", "Romania"),
    ("SE", "Sweden"), ("SI", "Slovenia"), ("SK", "Slovakia"), ("GB", "United Kingdom"),
    ("US", "United States"), ("JP", "Japan"), ("CN", "China"), ("CA", "Canada"),
    ("KR", "South Korea"), ("BR", "Brazil"), ("IN", "India"), ("MX", "Mexico"),
    ("RU", "Russia"), ("AU", "Australia"), ("CH", "Switzerland"), ("TR", "Türkiye"),
    ("TW", "Taiwan"), ("NO", "Norway"), ("ID", "Indonesia"), ("ZA", "South Africa"),
    ("WA", "RoW Asia-Pacific"), ("WL", "RoW America"), ("WE", "RoW Europe"),
    ("WF", "RoW Africa"), ("WM", "RoW Middle East"),
]
EU27_LABEL = ("EU27", "European Union (27)")
WORLD_LABEL = ("W", "Total 49 regions")

# ──────────────────────────────── Data reading ────────────────────────────────


def read_wdi(path: Path) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    """World Bank series in WDI format.

    Returns ({code: {year: value}}, {code: label}). All entities of the file are
    read, economies and aggregates alike; derive_perimeter sorts them out.
    """
    rows = list(csv.reader(path.open(encoding="utf-8-sig")))
    header = next(r for r in rows if r and r[0] == "Country Name")
    idx = {y: i for i, y in enumerate(header)}
    values, labels = {}, {}
    for r in rows:
        if len(r) <= idx["2025"] or r[1] == "Country Code":
            continue
        values[r[1]] = {y: float(r[idx[str(y)]])
                        for y in range(HIST_FIRST, HIST_LAST + 1) if r[idx[str(y)]].strip()}
        labels[r[1]] = r[0]
    return values, labels


def derive_perimeter(labels: dict[str, str], pop_wb: dict[str, dict[int, float]],
                     verbose: bool = True) -> list[str]:
    """Separate individual economies from the WDI aggregates.

    The WDI extractions mix economies and aggregates (World, income groups,
    regions, lending groups). The latter carry three-letter codes that do not
    belong to the ISO 3166-1 standard (WLD, HIC, EUU, SSF...), whereas all
    economies do, with the documented exception of NON_ISO_ECONOMIES. The
    sorting therefore relies on membership in the ISO 3166-1 table provided by
    country_converter, not on a list of aggregates, which would become obsolete
    with every revision of the nomenclature.

    The result is checked by a demographic closure test: the sum of the
    retained populations must approach the "World" aggregate of the file. An
    aggregate kept by mistake would produce a double count detected by this test.
    """
    iso3 = set(coco.CountryConverter().data["ISO3"]) | NON_ISO_ECONOMIES
    codes = sorted((c for c in labels if c in iso3), key=lambda c: labels[c])  # sorted by label

    world = pop_wb.get("WLD", {}).get(2019)
    if world:
        total = sum(pop_wb[c][2019] for c in codes if pop_wb.get(c, {}).get(2019))
        gap = total / world - 1
        if abs(gap) > PERIMETER_TOLERANCE:
            sys.exit(f"Inconsistent perimeter: the 2019 population of the {len(codes)} retained "
                     f"economies deviates by {gap:+.1%} from the \"World\" aggregate. "
                     f"Check the nomenclature of the WDI extraction.")
        if verbose:
            print(f"Perimeter: {len(codes)} economies out of {len(labels)} WDI entities "
                  f"({len(labels) - len(codes)} aggregates discarded) · "
                  f"2019 demographic closure: {gap:+.2%}")
    return codes


def read_ssp(path: Path, unit: str):
    """SSP nodes in IAMC format. Returns nodes(scenario, name) -> [values] or None."""
    rows = list(csv.reader(path.open()))
    header = rows[0]
    idx = {int(y): i for i, y in enumerate(header) if y.strip().isdigit()}
    table = {(r[1], r[2]): r for r in rows[1:] if r[4] == unit}

    def nodes(scenario: str, name: str):
        hist, proj = table.get(("Historical Reference", name)), table.get((scenario, name))
        if hist is None or proj is None:
            return None
        raw = [hist[idx[NODES[0]]]] + [proj[idx[y]] for y in NODES[1:]]
        return [float(v) for v in raw] if all(v.strip() for v in raw) else None

    names = sorted({r[2] for r in rows[1:]} - {"World", "European Union (R9)"})
    return nodes, names


def read_un(path: Path) -> dict[str, dict[int, float]]:
    """UN population (World Population Prospects 2024), in thousands on 1 July.

    Splices the "Estimates" sheet (1950-2023) and the medium variant (2024-2100).
    Returns {ISO3: {year: value}} for countries and territories. The population
    of France is supplemented with that of the overseas departments, counted
    separately by the UN but included in the World Bank perimeter.
    """
    wb = load_workbook(path, read_only=True, data_only=True)
    pop: dict[str, dict[int, float]] = {}
    for sheet in ("Estimates", "Medium variant"):
        for r in wb[sheet].iter_rows(min_row=18, values_only=True):
            if r[8] == "Country/Area" and r[5] and isinstance(r[10], int) and r[12] not in (None, ""):
                pop.setdefault(r[5], {})[r[10]] = float(r[12])
    for d in DROM:
        for y, v in pop.pop(d, {}).items():
            pop["FRA"][y] = pop["FRA"].get(y, 0.0) + v
    return pop


def classify(codes: list[str]) -> tuple[dict[str, str], set[str]]:
    """EXIOBASE 3 region of each economy and EU27 membership (country_converter)."""
    cc = coco.CountryConverter()
    region = dict(zip(codes, cc.convert(codes, src="ISO3", to="EXIO3", not_found=None)))
    region["CHI"] = "WE"                      # Channel Islands, not matched by EXIO3
    for d in DROM:
        region[d] = "FR"                      # overseas departments attached to France (WB perimeter)
    eu27 = set(cc.EU27as("ISO3")["ISO3"])
    return region, eu27

# ───────────────────────── Construction of annual series ──────────────────────


def ssp_entities(codes: list[str], ssp_names: list[str]) -> dict[str, str]:
    """Economy -> IIASA entity mapping (excluding the entities attached to France)."""
    cc = coco.CountryConverter()
    iso = dict(zip(ssp_names, cc.convert(ssp_names, src="regex", to="ISO3", not_found=None)))
    aggregated = {"GUF", "MYT", "TWN", "ESH"}  # French Guiana and Mayotte aggregated into FRA; no WB GDP for TWN/ESH
    return {code: name for name, code in iso.items() if code and code not in aggregated}


def build_series(nodes, wb: dict, scenario: str, codes: list[str],
                 region: dict[str, str], entity: dict[str, str]) -> tuple[dict, dict, dict]:
    """Annual 1990-2100 series by economy: observed up to t0, SSP beyond.

    Returns (series, status of the applied dynamics, anchor year t0).
    """
    own = {}
    for code in codes:
        if code == "FRA":
            parts = [nodes(scenario, n) for n in FR_PARTS]
            own[code] = [sum(v) for v in zip(*parts)] if all(parts) else None
        else:
            own[code] = nodes(scenario, entity[code]) if code in entity else None

    # Regional fallback aggregates, over the economies with their own projection only
    fallback = {
        g: [sum(own[c][j] for c in codes if own[c] and region[c] == g) for j in range(len(NODES))]
        for g in ROW_REGIONS
    }

    series, status, anchors = {}, {}, {}
    for code in codes:
        observed = wb.get(code, {})
        t0 = max(observed) if observed else None
        anchors[code] = t0

        if own[code]:
            nd, status[code] = own[code], STATUS_OWN
        elif t0 is not None:
            nd, status[code] = fallback[region[code]], STATUS_FALLBACK
        else:
            nd, status[code] = None, STATUS_NONE

        s = dict(observed)
        if t0 is not None and nd:
            def S(year: int, nd=nd) -> float:
                """SSP trajectory, geometrically interpolated between nodes."""
                if year % 5 == 0:
                    return nd[(year - NODES[0]) // 5]
                y0 = year - year % 5
                a, b = nd[(y0 - NODES[0]) // 5], nd[(y0 + 5 - NODES[0]) // 5]
                return a * (b / a) ** ((year - y0) / 5)

            for year in range(t0 + 1, LAST_YEAR + 1):
                s[year] = observed[t0] * S(year) / S(t0)
        series[code] = s

    return series, status, anchors

# ──────────────────────────── Share computation ───────────────────────────────


def country_shares(gdp: dict, pop: dict, included: list[str]) -> tuple[dict, dict]:
    """EPC (3 variants) and CTR (18 variants) shares by economy."""
    pop_mean = {c: [statistics.fmean(pop[c][y] for y in range(a, b + 1)) for _, a, b in PP]
                for c in included}

    gdpcap = {}
    for c in included:
        gdpcap[c] = []
        for _, a, b in GP:
            years = [y for y in range(a, b + 1) if gdp[c].get(y)]
            gdpcap[c].append(sum(gdp[c][y] for y in years) / sum(pop[c][y] for y in years))

    epc = {}
    for p in range(len(PP)):
        total = sum(pop_mean[c][p] for c in included)
        for c in included:
            epc.setdefault(c, []).append(pop_mean[c][p] / total)

    ctr = {}
    for p, g in COMBOS:
        ratios = {c: pop_mean[c][p] / gdpcap[c][g] for c in included}
        total = sum(ratios.values())
        for c in included:
            ctr.setdefault(c, []).append(ratios[c] / total)

    return epc, ctr


def un_shares(pop_un: dict, included: list[str]) -> tuple[dict, dict]:
    """EPC shares by economy (UN_PP periods) based on the UN population, independent of the SSPs.

    Returns (shares, mean population by economy and by period, in thousands).
    """
    missing = [c for c in included if c not in pop_un]
    if missing:
        sys.exit(f"Economies without UN population: {missing}")
    mean = {c: [statistics.fmean(pop_un[c][y] for y in range(a, b + 1)) for _, a, b in UN_PP]
            for c in included}
    totals = [sum(mean[c][p] for c in included) for p in range(len(UN_PP))]
    return {c: [mean[c][p] / totals[p] for p in range(len(UN_PP))] for c in included}, mean


def compute_all(verbose: bool = True) -> dict:
    """Country and regional shares, for the five SSPs and all variants."""
    gdp_wb, names = read_wdi(SRC["gdp_wb"])
    pop_wb, _ = read_wdi(SRC["pop_wb"])
    gdp_nodes, gdp_names = read_ssp(SRC["gdp_ssp"], UNIT_GDP_SSP)
    pop_nodes, pop_names = read_ssp(SRC["pop_ssp"], UNIT_POP_SSP)
    codes = derive_perimeter(names, pop_wb, verbose)
    region, eu27 = classify(codes)
    gdp_entity = ssp_entities(codes, gdp_names)
    pop_entity = ssp_entities(codes, pop_names)

    included = [c for c in codes if gdp_wb.get(c)]
    excluded = [c for c in codes if not gdp_wb.get(c)]
    if verbose:
        print(f"{len(included)} economies included · {len(excluded)} excluded "
              f"(no World Bank GDP 1990-{HIST_LAST})")

    epc, ctr, diagnostics = {}, {}, {}
    for k in SSPS:
        gdp, st_g, anchors = build_series(gdp_nodes, gdp_wb, k, codes, region, gdp_entity)
        pop, st_p, _ = build_series(pop_nodes, pop_wb, k, codes, region, pop_entity)
        epc[k], ctr[k] = country_shares(gdp, pop, included)
        diagnostics[k] = {"gdp_status": st_g, "pop_status": st_p, "anchor": anchors}
        if verbose:
            tot = sum(v[REF_COMBO] for v in ctr[k].values())
            print(f"  {k}: Σ CTR ref. = {tot:.12f}")

    pop_un = read_un(SRC["pop_un"])
    epc_un, _ = un_shares(pop_un, included)
    pop_mean = {c: [statistics.fmean(pop_un[c][y] for y in range(a, b + 1)) for _, a, b in POP_PP]
                for c in included}

    # Regional aggregation: sum of the country shares, variant by variant
    reg_epc = {k: {} for k in SSPS}
    reg_ctr = {k: {} for k in SSPS}
    reg_un, reg_pop = {}, {}
    for g in [g for g, _ in REGION_LABELS] + [EU27_LABEL[0]]:
        members = [c for c in included if (c in eu27 if g == EU27_LABEL[0] else region[c] == g)]
        reg_un[g] = [sum(epc_un[c][p] for c in members) for p in range(len(UN_PP))]
        reg_pop[g] = [sum(pop_mean[c][p] for c in members) for p in range(len(POP_PP))]
    for k in SSPS:
        for g, _ in REGION_LABELS:
            members = [c for c in included if region[c] == g]
            reg_epc[k][g] = [sum(epc[k][c][p] for c in members) for p in range(len(PP))]
            reg_ctr[k][g] = [sum(ctr[k][c][j] for c in members) for j in range(len(COMBOS))]
        members = [c for c in included if c in eu27]
        reg_epc[k]["EU27"] = [sum(epc[k][c][p] for c in members) for p in range(len(PP))]
        reg_ctr[k]["EU27"] = [sum(ctr[k][c][j] for c in members) for j in range(len(COMBOS))]

    counts = {g: (sum(1 for c in included if region[c] == g),
                  sum(1 for c in excluded if region[c] == g)) for g, _ in REGION_LABELS}
    counts["EU27"] = (sum(1 for c in included if c in eu27), sum(1 for c in excluded if c in eu27))

    return {"codes": codes, "names": names, "region": region, "eu27": eu27,
            "included": included, "excluded": excluded, "counts": counts,
            "epc": epc, "ctr": ctr, "reg_un": reg_un, "reg_pop": reg_pop,
            "reg_epc": reg_epc, "reg_ctr": reg_ctr,
            "diagnostics": diagnostics}

# ──────────────────────────────── xlsx output ─────────────────────────────────

DARK, MED, LIGHT, REF_FILL, CTRL = "1F3864", "2E5496", "D9E1F2", "FFF2CC", "DEEAF6"
PCT, RATIO = "0.0000%", "0.000"
NOT_AVAILABLE = "n.a."


def _font(**kw):
    return Font(name="Arial", size=kw.pop("size", 9), **kw)


def _head(ws, row, col, text, fill=DARK):
    cell = ws.cell(row, col, text)
    cell.font = _font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = Border(*[Side(style="thin", color="BFBFBF")] * 4)
    return cell


def _column_total(ws, col, first_row, last_row):
    """Sum of the numeric cells of a column (cells such as "n.a." are ignored)."""
    return sum(v for v in (ws.cell(r, col).value for r in range(first_row, last_row + 1))
               if isinstance(v, (int, float)))


def write_workbook(res: dict, out: Path) -> Path:
    wb = Workbook()

    # ── Summary sheet ──
    ws = wb.active
    ws.title = SHEET_REGIONS
    ws["A1"] = "Planetary budget shares by EXIOBASE 3 region – EPC and CTR principles"
    ws["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    ws.merge_cells("A1:P1")
    ws["A2"] = (
        f"Regional shares = sum of the shares computed at country level, variant by variant. "
        f"Reference: {REF_SSP}, EPC {PP[REF_P][0]} and CTR Pop {PP[REF_P][0]} × GDP {GP[REF_G][0]} "
        f"(setting without any projected data, hence independent of the scenario). "
        f"Min / mean / max: EPC over {len(PP) * len(SSPS)} values ({len(PP)} periods × {len(SSPS)} SSPs), "
        f"CTR over {len(COMBOS) * len(SSPS)} values ({len(COMBOS)} combinations × {len(SSPS)} SSPs). "
        f"The means add up to 100 %; the min and max do not, since they do not correspond to the same "
        f"setting from one region to another. Detail by variant: sheet {SHEET_VARIANTS}."
    )
    ws["A2"].font = _font(italic=True, color="595959")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:P2")
    ws.row_dimensions[2].height = 52

    headers = ["EXIOBASE code", "Region", "Countries included", "Countries excluded (no GDP)",
               f"EPC share ref. ({REF_SSP}, {PP[REF_P][0]})",
               f"CTR share ref. ({REF_SSP}, Pop {PP[REF_P][0]} × GDP {GP[REF_G][0]})",
               "CTR / EPC ratio ref.", "EPC min", "EPC mean", "EPC max",
               "CTR min", "CTR mean", "CTR max", "EU27"]
    for j, text in enumerate(headers, 1):
        _head(ws, 4, j, text)

    def stats(code):
        e = [res["reg_epc"][k][code][p] for k in SSPS for p in range(len(PP))]
        c = [res["reg_ctr"][k][code][j] for k in SSPS for j in range(len(COMBOS))]
        return e, c

    rows = REGION_LABELS + [EU27_LABEL]
    # The EU27 aggregate is separated from the total of the 49 regions by an empty row
    row_of = {code: 5 + i for i, (code, _) in enumerate(REGION_LABELS)}
    row_of["EU27"] = 5 + len(REGION_LABELS) + 2
    for code, label in rows:
        r = row_of[code]
        n_in, n_out = res["counts"][code]
        ws.cell(r, 1, code).font = _font()
        ws.cell(r, 2, label).font = _font()
        ws.cell(r, 3, n_in).font = _font()
        ws.cell(r, 4, n_out).font = _font()
        if n_in == 0:
            for j in (5, 6, 7, 8, 9, 10, 11, 12, 13):
                ws.cell(r, j, NOT_AVAILABLE).font = _font()
        else:
            e, c = stats(code)
            epc_ref = res["reg_epc"][REF_SSP][code][REF_P]
            ctr_ref = res["reg_ctr"][REF_SSP][code][REF_COMBO]
            values = [(5, epc_ref, PCT), (6, ctr_ref, PCT), (7, ctr_ref / epc_ref, RATIO),
                      (8, min(e), PCT), (9, statistics.fmean(e), PCT), (10, max(e), PCT),
                      (11, min(c), PCT), (12, statistics.fmean(c), PCT), (13, max(c), PCT)]
            for col, value, fmt in values:
                cell = ws.cell(r, col, value)
                cell.font = _font(bold=(col in (5, 6)))
                cell.number_format = fmt
                if col in (5, 6):
                    cell.fill = PatternFill("solid", fgColor=LIGHT)
        is_eu = code != "EU27" and any(c in res["eu27"] for c in res["codes"]
                                       if res["region"][c] == code)
        ws.cell(r, 14, "yes" if is_eu else "").font = _font()
        if code == "FR":
            for j in range(1, 15):
                ws.cell(r, j).fill = PatternFill("solid", fgColor=REF_FILL)
                ws.cell(r, j).font = _font(bold=True)

    # World row (W): totals written as values, not as =SUM() formulas: openpyxl
    # does not compute formulas, so they would have no cached value and would be
    # read as empty by pandas and by the external links of thresholds.xlsx.
    total_row = 5 + len(REGION_LABELS)
    ws.cell(total_row, 1, WORLD_LABEL[0]).font = _font(bold=True)  # World code, read by the by-region figure
    ws.cell(total_row, 2, WORLD_LABEL[1]).font = _font(bold=True)
    for col in (3, 4):
        cell = ws.cell(total_row, col, _column_total(ws, col, 5, total_row - 1))
        cell.font = _font(bold=True)
    for col in (5, 6, 9, 12):
        cell = ws.cell(total_row, col, _column_total(ws, col, 5, total_row - 1))
        cell.font = _font(bold=True)
        cell.number_format = PCT
        cell.fill = PatternFill("solid", fgColor=CTRL)

    note = row_of["EU27"] + 3
    ws.cell(note, 1, (
        "EU27 aggregate: sum of the shares of its 27 member states, for each variant; it is not a region of "
        "the partition and is not included in the total. Taiwan (TW): missing from the World Bank data, share "
        "cannot be computed. The regional min and max are the extremes of the regional shares over the "
        "variants, not the sum of the country min and max. Highlighted row: France."
    )).font = _font(italic=True, color="595959")
    ws.merge_cells(start_row=note, start_column=1, end_row=note + 2, end_column=14)
    ws.cell(note, 1).alignment = Alignment(wrap_text=True, vertical="top")

    ws.freeze_panes = "C5"
    ws.row_dimensions[4].height = 60
    for col, width in zip("ABCDEFGHIJKLMN", [9, 22, 8, 9, 12, 14, 9, 10, 10, 10, 10, 10, 10, 6]):
        ws.column_dimensions[col].width = width

    # ── Detailed sheet: all variants ──
    wv = wb.create_sheet(SHEET_VARIANTS)
    wv["A1"] = "Regional shares by variant and by scenario (sum of the country shares)"
    wv["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    wv["A2"] = ("One column per (scenario, variant) pair. Strictly historical variants "
                "(Pop 2019 × GDP 1990-2019 or 2019) are identical from one SSP to another. The first three columns "
                "(UN – EPC) rely on the UN population (WPP 2024, medium variant), independent of the SSPs; "
                f"the matching populations are in sheet {SHEET_POPULATION}.")
    wv["A2"].font = _font(italic=True, color="595959")

    _head(wv, 4, 1, "EXIOBASE code")
    _head(wv, 4, 2, "Region")
    col = 3
    layout = []
    for p, (plabel, _, _) in enumerate(UN_PP):
        _head(wv, 4, col, f"UN – EPC {plabel}", MED)
        layout.append((None, "un", p, col))
        col += 1
    for k in SSPS:
        for p, (plabel, _, _) in enumerate(PP):
            _head(wv, 4, col, f"{k} – EPC {plabel}", MED)
            layout.append((k, "epc", p, col))
            col += 1
        for j, (p, g) in enumerate(COMBOS):
            _head(wv, 4, col, f"{k} – CTR Pop {PP[p][0]} × GDP {GP[g][0]}", MED)
            layout.append((k, "ctr", j, col))
            col += 1

    for code, label in rows:
        r = row_of[code]
        wv.cell(r, 1, code).font = _font()
        wv.cell(r, 2, label).font = _font()
        n_in, _ = res["counts"][code]
        for k, kind, index, c in layout:
            if kind == "un":
                value = res["reg_un"][code][index]
            else:
                value = (res["reg_epc"] if kind == "epc" else res["reg_ctr"])[k][code][index]
            cell = wv.cell(r, c, value if n_in else NOT_AVAILABLE)
            cell.font = _font()
            if n_in:
                cell.number_format = PCT

    # World row (W): sum of the 49 regions
    wv.cell(total_row, 1, WORLD_LABEL[0]).font = _font(bold=True)
    wv.cell(total_row, 2, WORLD_LABEL[1]).font = _font(bold=True)
    for k, kind, index, c in layout:
        cell = wv.cell(total_row, c, _column_total(wv, c, 5, total_row - 1))
        cell.number_format = PCT
        cell.font = _font(bold=True)
        cell.fill = PatternFill("solid", fgColor=CTRL)

    wv.freeze_panes = "C5"
    wv.row_dimensions[4].height = 70
    wv.column_dimensions["A"].width = 9
    wv.column_dimensions["B"].width = 22
    for c in range(3, col):
        wv.column_dimensions[get_column_letter(c)].width = 13

    # ── UN population sheet: same rows as the share sheets, one column per period ──
    wp = wb.create_sheet(SHEET_POPULATION)
    wp["A1"] = "UN population by EXIOBASE region (thousands, mean over the period)"
    wp["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    wp["A2"] = ("World Population Prospects 2024: estimates up to 2023, medium variant from 2024. Sum over the "
                "economies included in the share calculations. Columns C-E: populations of the \"UN – EPC\" "
                f"shares of {SHEET_VARIANTS}; next columns: other periods used by PlaneFR.")
    wp["A2"].font = _font(italic=True, color="595959")
    _head(wp, 4, 1, "EXIOBASE code")
    _head(wp, 4, 2, "Region")
    for p, (plabel, _, _) in enumerate(POP_PP):
        _head(wp, 4, 3 + p, f"UN – Population {plabel} (thousands)", MED)
    for code, label in rows:
        r = row_of[code]
        wp.cell(r, 1, code).font = _font()
        wp.cell(r, 2, label).font = _font()
        for p in range(len(POP_PP)):
            cell = wp.cell(r, 3 + p, res["reg_pop"][code][p])
            cell.font = _font()
            cell.number_format = "#,##0"
    wp.cell(total_row, 1, WORLD_LABEL[0]).font = _font(bold=True)
    wp.cell(total_row, 2, WORLD_LABEL[1]).font = _font(bold=True)
    for p in range(len(POP_PP)):
        cell = wp.cell(total_row, 3 + p, _column_total(wp, 3 + p, 5, total_row - 1))
        cell.number_format = "#,##0"
        cell.font = _font(bold=True)
        cell.fill = PatternFill("solid", fgColor=CTRL)
    wp.freeze_panes = "C5"
    wp.row_dimensions[4].height = 45
    wp.column_dimensions["A"].width = 9
    wp.column_dimensions["B"].width = 22
    for p in range(len(POP_PP)):
        wp.column_dimensions[get_column_letter(3 + p)].width = 15

    # ── Diagnostics sheet ──
    wd = wb.create_sheet(SHEET_DIAGNOSTICS)
    wd["A1"] = "Diagnostics by economy (reference scenario)"
    wd["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    for j, text in enumerate(["Code", "Country", "EXIOBASE region", "EU27", "Included",
                              "GDP dynamics", "Population dynamics", "t0 anchor (GDP)",
                              "EPC share ref.", "CTR share ref."], 1):
        _head(wd, 3, j, text)
    diag = res["diagnostics"][REF_SSP]
    order = sorted(res["included"], key=lambda c: -res["ctr"][REF_SSP][c][REF_COMBO])
    order += sorted(res["excluded"], key=lambda c: res["names"][c])
    for i, code in enumerate(order):
        r = 4 + i
        inc = code in res["included"]
        for j, value in enumerate([code, res["names"][code], res["region"][code],
                                   "yes" if code in res["eu27"] else "",
                                   1 if inc else 0, diag["gdp_status"][code],
                                   diag["pop_status"][code], diag["anchor"][code]], 1):
            wd.cell(r, j, value).font = _font()
        if inc:
            for j, value in ((9, res["epc"][REF_SSP][code][REF_P]),
                             (10, res["ctr"][REF_SSP][code][REF_COMBO])):
                cell = wd.cell(r, j, value)
                cell.font = _font()
                cell.number_format = PCT
    wd.freeze_panes = "C4"
    wd.row_dimensions[3].height = 45
    for col, width in zip("ABCDEFGHIJ", [8, 32, 10, 6, 7, 22, 22, 11, 12, 12]):
        wd.column_dimensions[col].width = width

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out

# ─────────────────────────────────── Check ────────────────────────────────────


def check(res: dict, reference: Path) -> None:
    """Compare the regional shares with those of a reference workbook."""
    ws = load_workbook(reference, data_only=True)[SHEET_REGIONS]
    worst, n = 0.0, 0
    for r in list(range(5, 54)) + [56]:
        code = ws.cell(r, 1).value
        if code not in res["reg_epc"][REF_SSP] or not res["counts"][code][0]:
            continue
        pairs = [(ws.cell(r, 5).value, res["reg_epc"][REF_SSP][code][REF_P]),
                 (ws.cell(r, 6).value, res["reg_ctr"][REF_SSP][code][REF_COMBO])]
        e = [res["reg_epc"][k][code][p] for k in SSPS for p in range(len(PP))]
        c = [res["reg_ctr"][k][code][j] for k in SSPS for j in range(len(COMBOS))]
        pairs += list(zip([ws.cell(r, j).value for j in range(8, 14)],
                          [min(e), statistics.fmean(e), max(e),
                           min(c), statistics.fmean(c), max(c)]))
        for expected, obtained in pairs:
            if isinstance(expected, (int, float)) and expected:
                worst = max(worst, abs(obtained / expected - 1))
                n += 1
    print(f"Check on {n} regional values: maximum relative gap {worst:.1e}")
    if worst > 1e-9:
        sys.exit("Gap above tolerance: check the sources or the settings.")

# ──────────────────────────────────── main ────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                    help="output directory (default: directory of the script)")
    ap.add_argument("--check", type=Path, metavar="XLSX",
                    help="reference workbook used to check the regional shares")
    args = ap.parse_args()

    missing = [str(p) for p in SRC.values() if not p.exists()]
    if missing:
        sys.exit("Source files not found:\n  " + "\n  ".join(missing))

    res = compute_all()
    out = write_workbook(res, args.outdir / "budget_shares.xlsx")
    print(f"Written: {out.resolve()}")

    if args.check:
        check(res, args.check)


if __name__ == "__main__":
    main()
