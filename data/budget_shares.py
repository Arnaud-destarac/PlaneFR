#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PlanéFR – Parts de budget planétaire par région EXIOBASE 3
===========================================================

Calcule les parts d'allocation du budget planétaire selon deux principes d'équité,
pour les 49 régions EXIOBASE 3 et l'agrégat UE27, sous les cinq scénarios SSP et
pour l'ensemble des périodes de référence envisagées.

    EPC_s(P) = Pop_s(P) / Σ_s Pop_s(P)
    CTR_s(P, G) = [Pop_s(P) / GDPcap_s(G)] / Σ_s [Pop_s(P) / GDPcap_s(G)]

    avec GDPcap_s(G) = Σ_t PIB_t / Σ_t Pop_t sur la période G

La seconde écriture est la forme réduite de
CTR_s = (GDPcap_W / GDPcap_s) × (Pop_s / Pop_W) × α_AtP,
où α_AtP = Pop_W / (GDPcap_W × Σ_s Pop_s / GDPcap_s) : les termes mondiaux
s'annulent, les deux écritures sont strictement équivalentes.

Les parts régionales sont la somme des parts calculées au niveau pays, seule
agrégation cohérente avec le principe (l'indicateur Pop/GDPcap n'est pas additif).

Périodes
--------
    Démographiques (P) : 2019 · 2019-2050 · 2019-2100
    PIB (G)            : 1990-2019 · 1990-2050 · 2019 · 2019-2050 · 2019-2100 · 1990-2100

Soit 3 variantes EPC et 18 variantes CTR, pour chacun des 5 SSP.
Paramétrage de référence : SSP2, EPC 2019 et CTR Pop 2019 × PIB 1990-2019 — il ne
mobilise que des données observées et ne dépend donc d'aucun scénario.

Sources
-------
    Historique 1990-2025 : Banque mondiale, World Development Indicators
        PIB         NY.GDP.MKTP.PP.KD  (PPA, dollars internationaux constants 2021)
        Population  SP.POP.TOTL
    Projections 2026-2100 : base SSP de l'IIASA, mises à jour 2025
        PIB         modèle OECD ENV-Growth, variable GDP|PPP (USD 2017 PPA)
        Population  IIASA-Wittgenstein Centre (IIASA-WiC POP 2025)

Raccord des projections
-----------------------
Pour chaque économie, chaque variable et chaque scénario k :

    X_t = X_BM(t0) × S_k(t) / S_k(t0)     pour t > t0

où t0 est la dernière année observée par la Banque mondiale et S_k la trajectoire
SSP interpolée géométriquement entre jalons quinquennaux. Le niveau reste celui de
la Banque mondiale ; seule la dynamique est empruntée aux SSP, sans conversion de
base PPA — le rapport S_k(t)/S_k(t0) y est invariant.

Cohérence PIB / population : dans chaque scénario, GDPcap_s est calculé avec la
population du même SSP que le PIB, le PIB SSP ayant été projeté avec la démographie
SSP correspondante.

Remarques d'implémentation
--------------------------
  - France : la dynamique SSP est calculée sur la somme France + Guyane + Mayotte,
    entités distinctes dans la base IIASA, afin d'approcher le périmètre Banque
    mondiale qui inclut les DROM. Guadeloupe, Martinique et La Réunion n'ont pas
    d'entité propre chez l'IIASA et ne sont pas distinguées.
  - Repli régional : les économies sans projection SSP propre suivent la dynamique
    SSP agrégée de leur région EXIOBASE, calculée sur les économies de la région
    dotées d'une projection propre.
  - Périmètre : les économies individuelles sont isolées des agrégats du WDI par
    appartenance à la norme ISO 3166-1, et non par une liste d'agrégats figée ; le
    résultat est contrôlé par bouclage démographique sur l'agrégat « World ».
  - Économies exclues : celles sans PIB Banque mondiale sur 1990-2025 sont écartées
    de tous les calculs, EPC compris — EPC et CTR portent sur le même ensemble.
  - Les économies à couverture partielle voient GDPcap_s calculé sur les seules
    années disponibles de chaque période (sommes de population restreintes aux
    années avec PIB).

Usage
-----
    python3 budget_shares_regions.py [--outdir DIR] [--check CLASSEUR.xlsx]

Le script est autonome : les quatre extractions sources suffisent à reproduire
l'intégralité des résultats, sans fichier intermédiaire.

Dépendances : openpyxl, country_converter (pip install openpyxl country_converter)
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
from pathlib import Path

logging.disable(logging.WARNING)  # country_converter est verbeux sur les non-appariés

import country_converter as coco
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ─────────────────────────────── Configuration ────────────────────────────────

DATA = Path(r"C:\Users\Arnaud\Documents\CIRED\PlaneFR\Code\Module_PlaneFR\data")
SRC = {
    "gdp_bm": DATA / "Budget share" / "GDP PPP BM.csv",
    "pop_bm": DATA / "Budget share" / "Population BM.csv",
    "gdp_ssp": DATA / "Budget share" / "GDP PPP SSPs.csv",
    "pop_ssp": DATA / "Budget share" / "Population SSPs.csv",
    "pop_un": DATA / "Budget share" / "Population ONU.xlsx"
}


# Économies des extractions WDI dont le code n'appartient pas à la norme ISO 3166-1
# et que le filtre du périmètre (cf. derive_perimeter) retiendrait donc à tort pour
# des agrégats. CHI regroupe Guernesey et Jersey, économies non distinguées par le WDI.
NON_ISO_ECONOMIES = {"CHI"}

# Tolérance du contrôle de bouclage démographique : écart admis entre la somme des
# populations retenues et l'agrégat « World » du WDI (économies à population non
# renseignée, arrondis). Un agrégat régional conservé par erreur ferait exploser cet écart.
PERIMETER_TOLERANCE = 0.02

SSPS = ["SSP1", "SSP2", "SSP3", "SSP4", "SSP5"]
NODES = list(range(2020, 2101, 5))          # jalons SSP : 2020 (Historical Reference), puis scénario
HIST_FIRST, HIST_LAST = 1990, 2025          # couverture Banque mondiale
LAST_YEAR = 2100

PP = [("2019", 2019, 2019), ("2019-2050", 2019, 2050), ("2019-2100", 2019, 2100)]
# Périodes démographiques des parts EPC fondées sur la population ONU
UN_PP = [("2019", 2019, 2019), ("2050", 2050, 2050), ("2019-2100", 2019, 2100)]
GP = [("1990-2019", 1990, 2019), ("1990-2050", 1990, 2050), ("2019", 2019, 2019),
      ("2019-2050", 2019, 2050), ("2019-2100", 2019, 2100), ("1990-2100", 1990, 2100)]
COMBOS = [(p, g) for p in range(len(PP)) for g in range(len(GP))]

REF_SSP = "SSP2"
REF_P, REF_G = 0, 0                         # Pop 2019 × PIB 1990-2019
REF_COMBO = REF_P * len(GP) + REF_G

UNIT_GDP_SSP = "billion USD_2017/yr"
UNIT_POP_SSP = "million"

FR_PARTS = ["France", "French Guiana", "Mayotte"]
DROM = ["GLP", "MTQ", "GUF", "REU", "MYT"]
ROW_REGIONS = ["WA", "WE", "WF", "WL", "WM"]  # régions « Rest of World », supports du repli

REGION_LABELS = [
    ("AT", "Autriche"), ("BE", "Belgique"), ("BG", "Bulgarie"), ("CY", "Chypre"),
    ("CZ", "Tchéquie"), ("DE", "Allemagne"), ("DK", "Danemark"), ("EE", "Estonie"),
    ("ES", "Espagne"), ("FI", "Finlande"), ("FR", "France"), ("GR", "Grèce"),
    ("HR", "Croatie"), ("HU", "Hongrie"), ("IE", "Irlande"), ("IT", "Italie"),
    ("LT", "Lituanie"), ("LU", "Luxembourg"), ("LV", "Lettonie"), ("MT", "Malte"),
    ("NL", "Pays-Bas"), ("PL", "Pologne"), ("PT", "Portugal"), ("RO", "Roumanie"),
    ("SE", "Suède"), ("SI", "Slovénie"), ("SK", "Slovaquie"), ("GB", "Royaume-Uni"),
    ("US", "États-Unis"), ("JP", "Japon"), ("CN", "Chine"), ("CA", "Canada"),
    ("KR", "Corée du Sud"), ("BR", "Brésil"), ("IN", "Inde"), ("MX", "Mexique"),
    ("RU", "Russie"), ("AU", "Australie"), ("CH", "Suisse"), ("TR", "Turquie"),
    ("TW", "Taïwan"), ("NO", "Norvège"), ("ID", "Indonésie"), ("ZA", "Afrique du Sud"),
    ("WA", "RoW Asie-Pacifique"), ("WL", "RoW Amérique"), ("WE", "RoW Europe"),
    ("WF", "RoW Afrique"), ("WM", "RoW Moyen-Orient"),
]

# ──────────────────────────── Lecture des données ─────────────────────────────


def read_wdi(path: Path) -> tuple[dict[str, dict[int, float]], dict[str, str]]:
    """Série Banque mondiale au format WDI.

    Renvoie ({code: {année: valeur}}, {code: libellé}). Toutes les entités du fichier
    sont lues, économies comme agrégats ; le tri est opéré par derive_perimeter.
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


def derive_perimeter(labels: dict[str, str], pop_bm: dict[str, dict[int, float]],
                     verbose: bool = True) -> list[str]:
    """Isole les économies individuelles des agrégats du WDI.

    Les extractions WDI mêlent économies et agrégats (World, groupes de revenu,
    régions, groupes de prêt). Ces derniers portent des codes à trois lettres qui
    n'appartiennent pas à la norme ISO 3166-1 (WLD, HIC, EUU, SSF…), alors que les
    économies en relèvent toutes, à l'exception documentée de NON_ISO_ECONOMIES.
    Le tri repose donc sur l'appartenance à la table ISO 3166-1 fournie par
    country_converter, et non sur une liste d'agrégats, qui deviendrait caduque à
    chaque révision de la nomenclature.

    Le résultat est contrôlé par bouclage démographique : la somme des populations
    retenues doit approcher l'agrégat « World » du fichier. Un agrégat conservé par
    erreur produirait un double compte détecté par ce test.
    """
    iso3 = set(coco.CountryConverter().data["ISO3"]) | NON_ISO_ECONOMIES
    codes = sorted((c for c in labels if c in iso3), key=lambda c: labels[c])  # tri par libellé

    world = pop_bm.get("WLD", {}).get(2019)
    if world:
        total = sum(pop_bm[c][2019] for c in codes if pop_bm.get(c, {}).get(2019))
        gap = total / world - 1
        if abs(gap) > PERIMETER_TOLERANCE:
            sys.exit(f"Périmètre incohérent : Σ population 2019 des {len(codes)} économies "
                     f"retenues s'écarte de {gap:+.1%} de l'agrégat « World ». "
                     f"Vérifier la nomenclature de l'extraction WDI.")
        if verbose:
            print(f"Périmètre : {len(codes)} économies sur {len(labels)} entités WDI "
                  f"({len(labels) - len(codes)} agrégats écartés) · "
                  f"bouclage démographique 2019 : {gap:+.2%}")
    return codes


def read_ssp(path: Path, unit: str):
    """Jalons SSP au format IAMC. Renvoie nodes(scenario, nom) -> [valeurs] ou None."""
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
    """Population ONU (World Population Prospects 2024), en milliers au 1er juillet.

    Raccorde la feuille « Estimates » (1950-2023) et la variante médiane (2024-2100).
    Renvoie {ISO3: {année: valeur}} pour les pays et territoires. La population de la
    France est complétée de celle des DROM, comptés à part par l'ONU mais inclus dans
    le périmètre Banque mondiale.
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
    """Région EXIOBASE 3 par économie et appartenance à l'UE27 (country_converter)."""
    cc = coco.CountryConverter()
    region = dict(zip(codes, cc.convert(codes, src="ISO3", to="EXIO3", not_found=None)))
    region["CHI"] = "WE"                      # îles Anglo-Normandes, non appariées par EXIO3
    for d in DROM:
        region[d] = "FR"                      # DROM rattachés à la France (périmètre BM)
    ue27 = set(cc.EU27as("ISO3")["ISO3"])
    return region, ue27

# ───────────────────── Construction des séries annuelles ──────────────────────


def ssp_entities(codes: list[str], ssp_names: list[str]) -> dict[str, str]:
    """Correspondance économie -> entité IIASA (hors entités rattachées à la France)."""
    cc = coco.CountryConverter()
    iso = dict(zip(ssp_names, cc.convert(ssp_names, src="regex", to="ISO3", not_found=None)))
    aggregated = {"GUF", "MYT", "TWN", "ESH"}  # Guyane et Mayotte agrégées à FRA ; pas de PIB BM pour TWN/ESH
    return {code: name for name, code in iso.items() if code and code not in aggregated}


def build_series(nodes, bm: dict, scenario: str, codes: list[str],
                 region: dict[str, str], entity: dict[str, str]) -> tuple[dict, dict, dict]:
    """Séries annuelles 1990-2100 par économie : observé jusqu'à t0, SSP au-delà.

    Renvoie (séries, statut de la dynamique appliquée, année d'ancrage t0).
    """
    own = {}
    for code in codes:
        if code == "FRA":
            parts = [nodes(scenario, n) for n in FR_PARTS]
            own[code] = [sum(v) for v in zip(*parts)] if all(parts) else None
        else:
            own[code] = nodes(scenario, entity[code]) if code in entity else None

    # Agrégats régionaux de repli, sur les seules économies à projection propre
    fallback = {
        g: [sum(own[c][j] for c in codes if own[c] and region[c] == g) for j in range(len(NODES))]
        for g in ROW_REGIONS
    }

    series, status, anchors = {}, {}, {}
    for code in codes:
        observed = bm.get(code, {})
        t0 = max(observed) if observed else None
        anchors[code] = t0

        if own[code]:
            nd, status[code] = own[code], "SSP pays"
        elif t0 is not None:
            nd, status[code] = fallback[region[code]], "Repli région EXIOBASE"
        else:
            nd, status[code] = None, "Aucune"

        s = dict(observed)
        if t0 is not None and nd:
            def S(year: int, nd=nd) -> float:
                """Trajectoire SSP, interpolée géométriquement entre jalons."""
                if year % 5 == 0:
                    return nd[(year - NODES[0]) // 5]
                y0 = year - year % 5
                a, b = nd[(y0 - NODES[0]) // 5], nd[(y0 + 5 - NODES[0]) // 5]
                return a * (b / a) ** ((year - y0) / 5)

            for year in range(t0 + 1, LAST_YEAR + 1):
                s[year] = observed[t0] * S(year) / S(t0)
        series[code] = s

    return series, status, anchors

# ──────────────────────────── Calcul des parts ────────────────────────────────


def country_shares(gdp: dict, pop: dict, included: list[str]) -> tuple[dict, dict]:
    """Parts EPC (3 variantes) et CTR (18 variantes) par économie."""
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


def un_shares(pop_un: dict, included: list[str]) -> dict:
    """Parts EPC par économie (périodes UN_PP) d'après la population ONU, indépendantes des SSP."""
    missing = [c for c in included if c not in pop_un]
    if missing:
        sys.exit(f"Économies sans population ONU : {missing}")
    mean = {c: [statistics.fmean(pop_un[c][y] for y in range(a, b + 1)) for _, a, b in UN_PP]
            for c in included}
    totals = [sum(mean[c][p] for c in included) for p in range(len(UN_PP))]
    return {c: [mean[c][p] / totals[p] for p in range(len(UN_PP))] for c in included}


def compute_all(verbose: bool = True) -> dict:
    """Parts pays et régionales, pour les cinq SSP et toutes les variantes."""
    gdp_bm, names = read_wdi(SRC["gdp_bm"])
    pop_bm, _ = read_wdi(SRC["pop_bm"])
    gdp_nodes, gdp_names = read_ssp(SRC["gdp_ssp"], UNIT_GDP_SSP)
    pop_nodes, pop_names = read_ssp(SRC["pop_ssp"], UNIT_POP_SSP)
    codes = derive_perimeter(names, pop_bm, verbose)
    region, ue27 = classify(codes)
    gdp_entity = ssp_entities(codes, gdp_names)
    pop_entity = ssp_entities(codes, pop_names)

    included = [c for c in codes if gdp_bm.get(c)]
    excluded = [c for c in codes if not gdp_bm.get(c)]
    if verbose:
        print(f"{len(included)} économies incluses · {len(excluded)} exclues "
              f"(aucun PIB Banque mondiale 1990-{HIST_LAST})")

    epc, ctr, diagnostics = {}, {}, {}
    for k in SSPS:
        gdp, st_g, anchors = build_series(gdp_nodes, gdp_bm, k, codes, region, gdp_entity)
        pop, st_p, _ = build_series(pop_nodes, pop_bm, k, codes, region, pop_entity)
        epc[k], ctr[k] = country_shares(gdp, pop, included)
        diagnostics[k] = {"statut_pib": st_g, "statut_pop": st_p, "ancrage": anchors}
        if verbose:
            tot = sum(v[REF_COMBO] for v in ctr[k].values())
            print(f"  {k} : Σ CTR réf. = {tot:.12f}")

    epc_un = un_shares(read_un(SRC["pop_un"]), included)

    # Agrégation régionale : somme des parts pays, variante par variante
    reg_epc = {k: {} for k in SSPS}
    reg_ctr = {k: {} for k in SSPS}
    reg_un = {}
    for g in [g for g, _ in REGION_LABELS] + ["EU27"]:
        members = [c for c in included if (c in ue27 if g == "EU27" else region[c] == g)]
        reg_un[g] = [sum(epc_un[c][p] for c in members) for p in range(len(UN_PP))]
    for k in SSPS:
        for g, _ in REGION_LABELS:
            members = [c for c in included if region[c] == g]
            reg_epc[k][g] = [sum(epc[k][c][p] for c in members) for p in range(len(PP))]
            reg_ctr[k][g] = [sum(ctr[k][c][j] for c in members) for j in range(len(COMBOS))]
        members = [c for c in included if c in ue27]
        reg_epc[k]["EU27"] = [sum(epc[k][c][p] for c in members) for p in range(len(PP))]
        reg_ctr[k]["EU27"] = [sum(ctr[k][c][j] for c in members) for j in range(len(COMBOS))]

    counts = {g: (sum(1 for c in included if region[c] == g),
                  sum(1 for c in excluded if region[c] == g)) for g, _ in REGION_LABELS}
    counts["EU27"] = (sum(1 for c in included if c in ue27), sum(1 for c in excluded if c in ue27))

    return {"codes": codes, "names": names, "region": region, "ue27": ue27,
            "included": included, "excluded": excluded, "counts": counts,
            "epc": epc, "ctr": ctr, "reg_un": reg_un, "reg_epc": reg_epc, "reg_ctr": reg_ctr,
            "diagnostics": diagnostics}

# ──────────────────────────────── Sortie xlsx ─────────────────────────────────

DARK, MED, LIGHT, REF_FILL, CTRL = "1F3864", "2E5496", "D9E1F2", "FFF2CC", "DEEAF6"
PCT, RATIO = "0.0000%", "0.000"


def _font(**kw):
    return Font(name="Arial", size=kw.pop("size", 9), **kw)


def _head(ws, row, col, text, fill=DARK):
    cell = ws.cell(row, col, text)
    cell.font = _font(bold=True, color="FFFFFF")
    cell.fill = PatternFill("solid", fgColor=fill)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border = Border(*[Side(style="thin", color="BFBFBF")] * 4)
    return cell


def write_workbook(res: dict, out: Path) -> Path:
    wb = Workbook()

    # ── Feuille de synthèse ──
    ws = wb.active
    ws.title = "Budget_shares_regions"
    ws["A1"] = "Parts du budget planétaire par région EXIOBASE 3 – principes EPC et CTR"
    ws["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    ws.merge_cells("A1:P1")
    ws["A2"] = (
        f"Parts régionales = somme des parts calculées au niveau pays, variante par variante. "
        f"Référence : {REF_SSP}, EPC {PP[REF_P][0]} et CTR Pop {PP[REF_P][0]} × PIB {GP[REF_G][0]} "
        f"(paramétrage sans aucune donnée projetée, donc indépendant du scénario). "
        f"Min / moyenne / max : EPC sur {len(PP) * len(SSPS)} valeurs ({len(PP)} périodes × {len(SSPS)} SSP), "
        f"CTR sur {len(COMBOS) * len(SSPS)} valeurs ({len(COMBOS)} combinaisons × {len(SSPS)} SSP). "
        f"Les moyennes somment à 100 % ; les min et max, non, car ils ne correspondent pas au même "
        f"paramétrage d'une région à l'autre. Détail par variante : feuille Budget_shares_variantes."
    )
    ws["A2"].font = _font(italic=True, color="595959")
    ws["A2"].alignment = Alignment(wrap_text=True, vertical="top")
    ws.merge_cells("A2:P2")
    ws.row_dimensions[2].height = 52

    headers = ["Code EXIOBASE", "Région", "Nb pays inclus", "Nb pays exclus (sans PIB)",
               f"Part EPC ref. ({REF_SSP}, {PP[REF_P][0]})",
               f"Part CTR ref. ({REF_SSP}, Pop {PP[REF_P][0]} × PIB {GP[REF_G][0]})",
               "Rapport CTR / EPC ref.", "EPC min", "EPC moyenne", "EPC max",
               "CTR min", "CTR moyenne", "CTR max", "UE27"]
    for j, text in enumerate(headers, 1):
        _head(ws, 4, j, text)

    def stats(code):
        e = [res["reg_epc"][k][code][p] for k in SSPS for p in range(len(PP))]
        c = [res["reg_ctr"][k][code][j] for k in SSPS for j in range(len(COMBOS))]
        return e, c

    rows = REGION_LABELS + [("EU27", "Union européenne (27)")]
    # L'agrégat UE27 est séparé du total des 49 régions par une ligne vide
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
                ws.cell(r, j, "n.d.").font = _font()
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
        is_ue = code != "EU27" and any(c in res["ue27"] for c in res["codes"]
                                       if res["region"][c] == code)
        ws.cell(r, 14, "oui" if is_ue else "").font = _font()
        if code == "FR":
            for j in range(1, 15):
                ws.cell(r, j).fill = PatternFill("solid", fgColor=REF_FILL)
                ws.cell(r, j).font = _font(bold=True)

    total_row = 5 + len(REGION_LABELS)
    ws.cell(total_row, 1, "W").font = _font(bold=True)  # code Monde, lu par la figure par région
    ws.cell(total_row, 2, "Total 49 régions").font = _font(bold=True)
    for col in (3, 4):
        cell = ws.cell(total_row, col, f"=SUM({get_column_letter(col)}5:{get_column_letter(col)}{total_row - 1})")
        cell.font = _font(bold=True)
    for col in (5, 6, 9, 12):
        cell = ws.cell(total_row, col, f"=SUM({get_column_letter(col)}5:{get_column_letter(col)}{total_row - 1})")
        cell.font = _font(bold=True)
        cell.number_format = PCT
        cell.fill = PatternFill("solid", fgColor=CTRL)

    note = row_of["EU27"] + 3
    ws.cell(note, 1, (
        "Agrégat UE27 : somme des parts de ses 27 États membres, pour chaque variante ; il n'est pas une région "
        "de la partition et n'entre pas dans le total. Taïwan (TW) : absent des données Banque mondiale, part non "
        "calculable. Les min et max régionaux sont les extrêmes des parts régionales sur les variantes, et non la "
        "somme des min et max des pays. Ligne surlignée : France."
    )).font = _font(italic=True, color="595959")
    ws.merge_cells(start_row=note, start_column=1, end_row=note + 2, end_column=14)
    ws.cell(note, 1).alignment = Alignment(wrap_text=True, vertical="top")

    ws.freeze_panes = "C5"
    ws.row_dimensions[4].height = 60
    for col, width in zip("ABCDEFGHIJKLMN", [9, 22, 8, 9, 12, 14, 9, 10, 10, 10, 10, 10, 10, 6]):
        ws.column_dimensions[col].width = width

    # ── Feuille détaillée : toutes les variantes ──
    wv = wb.create_sheet("Budget_shares_variantes")
    wv["A1"] = "Parts régionales par variante et par scénario (somme des parts pays)"
    wv["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    wv["A2"] = ("Une colonne par couple (scénario, variante). Les variantes strictement historiques "
                "(Pop 2019 × PIB 1990-2019 ou 2019) sont identiques d'un SSP à l'autre. Les trois premières colonnes "
                "(UN – EPC) reposent sur la population ONU (WPP 2024, variante médiane), indépendante des SSP.")
    wv["A2"].font = _font(italic=True, color="595959")

    _head(wv, 4, 1, "Code EXIOBASE")
    _head(wv, 4, 2, "Région")
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
            _head(wv, 4, col, f"{k} – CTR Pop {PP[p][0]} × PIB {GP[g][0]}", MED)
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
            cell = wv.cell(r, c, value if n_in else "n.d.")
            cell.font = _font()
            if n_in:
                cell.number_format = PCT

    wv.freeze_panes = "C5"
    wv.row_dimensions[4].height = 70
    wv.column_dimensions["A"].width = 9
    wv.column_dimensions["B"].width = 22
    for c in range(3, col):
        wv.column_dimensions[get_column_letter(c)].width = 13

    # ── Feuille de diagnostic ──
    wd = wb.create_sheet("Diagnostic_pays")
    wd["A1"] = "Diagnostic par économie (scénario de référence)"
    wd["A1"].font = Font(name="Arial", size=13, bold=True, color=DARK)
    for j, text in enumerate(["Code", "Pays", "Région EXIOBASE", "UE27", "Inclus",
                              "Dynamique PIB", "Dynamique population", "Ancrage t0 (PIB)",
                              f"Part EPC réf.", f"Part CTR réf."], 1):
        _head(wd, 3, j, text)
    diag = res["diagnostics"][REF_SSP]
    order = sorted(res["included"], key=lambda c: -res["ctr"][REF_SSP][c][REF_COMBO])
    order += sorted(res["excluded"], key=lambda c: res["names"][c])
    for i, code in enumerate(order):
        r = 4 + i
        inc = code in res["included"]
        for j, value in enumerate([code, res["names"][code], res["region"][code],
                                   "oui" if code in res["ue27"] else "",
                                   1 if inc else 0, diag["statut_pib"][code],
                                   diag["statut_pop"][code], diag["ancrage"][code]], 1):
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

# ─────────────────────────────── Vérification ─────────────────────────────────


def check(res: dict, reference: Path) -> None:
    """Compare les parts régionales à celles d'un classeur de référence."""
    ws = load_workbook(reference, data_only=True)["Parts_regions_EXIOBASE"]
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
    print(f"Vérification sur {n} valeurs régionales : écart relatif maximal {worst:.1e}")
    if worst > 1e-9:
        sys.exit("Écart supérieur à la tolérance : vérifier les sources ou le paramétrage.")

# ──────────────────────────────────── main ────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent,
                    help="répertoire de sortie (défaut : répertoire du script)")
    ap.add_argument("--check", type=Path, metavar="XLSX",
                    help="classeur de référence pour vérifier les parts régionales")
    args = ap.parse_args()

    missing = [str(p) for p in SRC.values() if not p.exists()]
    if missing:
        sys.exit("Fichiers source introuvables :\n  " + "\n  ".join(missing))

    res = compute_all()
    out = write_workbook(res, args.outdir / "budget_shares.xlsx")
    print(f"Écrit : {out.resolve()}")

    if args.check:
        check(res, args.check)


if __name__ == "__main__":
    main()
