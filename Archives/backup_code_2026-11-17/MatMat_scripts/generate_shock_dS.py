"""
Génère dS, la variation relative de S (= F_x_dom / x) entre 2020 et 2050, pour
chaque scénario SSP et chaque extension renseignés dans
Module_PlaneFR/data/SSP/Matching_extensions_EXIOBASE_IMAGE.xlsx.

Une feuille par scénario (SSP1, SSP2, ...). Chaque ligne de la feuille donne :
    - 'Domaine'            : l'extension (Air_emissions -> air_emissions, ...)
    - 'Extension EXIOBASE' : la (ou les) ligne(s) de S_x_dom à choquer
    - 'Secteurs EXIOBASE'  : les secteurs concernés, séparés par ';'
                             (case vide = tous les secteurs)
    - puis, par région IMAGE agrégée, trois sous-colonnes dont 'S 2050/2020'.

    dS = S 2050/2020 - 1

Comptes de base : tous les éléments de structure des chocs (S_x_dom, A,
detail_levels.xlsx, base_year des info.json) sont lus dans les comptes Monde
auxquels le moteur applique ces chocs, c'est-à-dire path_in.accounts de
settings_engine_SSPs.json (BASE_ACCOUNTS_DIR, ex. Outputs/World/Base_year_2015 ;
cf. compute_ds_x_dom.get_ssp_base_accounts_dir). Changer d'année de base se fait
donc en un seul endroit, ce fichier de settings, puis en relançant ce script.

dS a le même format que le S_x_dom.pkl des comptes de base (indicateurs en ligne,
(region, sector) en colonne) et est initialisé à 0 ; seules les cellules
couvertes par le fichier de matching sont renseignées. Les régions EXIOBASE
sont rattachées aux régions IMAGE via la feuille 'regions' de bridge_SSPs.xlsx.

Correspondance des lignes : si 'Extension EXIOBASE' est exactement un
indicateur de S_x_dom, seul celui-ci est choqué ; sinon le libellé est découpé
sur ' - ' et tous les indicateurs contenant chacun des segments (sans tenir
compte de la casse) sont choqués, ex. 'Cropland - Wheat' -> 'Cropland -
cropped area - Wheat' et 'Cropland - fallowed area - Wheat'. Quand plusieurs
lignes du fichier couvrent la même cellule, la plus spécifique (celle qui
couvre le moins d'indicateurs) l'emporte. Cas particulier : les lignes
'Cropland' (SECTOR_IN_ROW_NAME) ne s'appliquent qu'aux indicateurs contenant
'Cropland' ET l'un des secteurs de 'Secteurs EXIOBASE', ex. 'Cropland' +
'Cattle' -> 'Cropland - fallowed area-Cattle'.

Secteurs : le choc ne s'applique qu'aux secteurs ayant une valeur non nulle
(dans au moins une région) pour l'indicateur considéré dans le S_x_dom des
comptes de base, en plus du filtre éventuel de 'Secteurs EXIOBASE'.

Régions : chaque région EXIOBASE est rattachée à une seule région IMAGE
agrégée ; toutes les régions EXIOBASE d'une même région IMAGE reçoivent le
même dS (ex. FR, DE, GB, CH, NO -> WEU).

dS_x_dom (.pkl uniquement) est écrit dans
MatMat/matmat-ademe/data/4-PlaneFR/2-shocks/3-SSPs_2050/<feuille>_2050/extensions/<ext>/
(dossiers créés s'ils n'existent pas), avec une copie du detail_levels.xlsx de
l'extension dans les comptes de base. Le dossier <feuille>_2050/system/ reçoit de
même une copie du detail_levels.xlsx du system des comptes de base. Les info.json
(system et extensions) portent le base_year des comptes de base et sont mis à
jour à chaque exécution.

Chocs sur A : une ligne dont la colonne 'Domaine' contient une flèche
(A_SHOCK_MARKER, ex. 'Cultures → élevage') est un choc sur les coefficients
techniques A, et non sur S. 'Extension EXIOBASE' donne alors les produits
EXIOBASE en ligne de A (séparés par ';', toutes régions d'origine confondues) et
'Secteurs EXIOBASE' les secteurs en colonne. Chaque colonne (region, sector)
reçoit les valeurs IMAGE de la région IMAGE de rattachement de sa région (région
qui consomme l'intrant). Le choc est calculé sur le A des comptes de base, en
euros, dans cet ordre :

1. Domaines à facteur unique (tous les domaines "→" sauf les deux suivants, ex.
   'Énergie → Agriculture', 'Cultures → élevage', 'Combustibles → centrales') :
   A' = A x S 2050/2020 de la feuille du scénario.
2. Groupes d'utilisateurs ('Énergie → <groupe>' à plusieurs lignes, une par
   vecteur énergétique ; FE = somme des variables IMAGE 'Numérateur F' des
   vecteurs qui en ont une, act = 'Dénominateur x') :
   - intensité I = [FE/act 2050] / [FE/act 2020] (1 si non calculable, ou si
     l'intensité 2020 de la région est inférieure à MIN_RELATIVE_INTENSITY fois
     celle du Monde) ; énergie totale de la colonne E' = I x E, E = somme des
     coefficients des produits des vecteurs du groupe ;
   - parts : Δv = part de v dans FE en 2050 - part en 2020 (0 pour un vecteur
     sans variable IMAGE), s'v = max(0, sv + Δv) renormalisé, sv = part en euros
     de v dans E ; total du vecteur = E' x s'v. Pas appliqué aux HEAVY_GROUPS
     (cas central : chaque vecteur est seulement multiplié par I).
3. Mix électrique (ELECTRICITY_MIX_DOMAIN, une ligne par filière) : même calcul
   de parts sur les filières, total électricité des filières de la colonne
   conservé. Les produits électriques hors filières (Electricity nec,
   transport/distribution...) ne sont pas modifiés par cette étape.

Dans un vecteur (ou une filière), les coefficients sont mis à l'échelle au
prorata (poids des produits et des régions d'origine conservés).

Le moteur applique dA de façon relative (A = A0 x (1 + dA)) : un coefficient
nul dans A0 le reste. dA = A'/A - 1 là où A > 0, 0 ailleurs. Pour que A' soit
exactement atteint, un vecteur (ou une filière) absent d'une colonne ne peut
pas y apparaître : les parts s'v sont renormalisées sur les seuls vecteurs
présents dans la colonne, c'est-à-dire de part >= MIN_VECTOR_SHARE (le total
E', ou le total électricité, est conservé ; un vecteur de part résiduelle est
mis à 0).
dA est écrit dans <feuille>_2050/system/dA.pkl (pas de .xlsx : 9800 x 9800
cellules).

Garde-fou : avant l'écriture de dA, le rayon spectral de A' est calculé.
L'inverse de Leontief (I - A')^-1 n'est fini et positif que si ce rayon est < 1 ;
au-delà de MAX_SPECTRAL_RADIUS (empreintes énormes ou de signe quelconque), aucun
choc sur A n'est écrit pour ce scénario. Les colonnes dont la somme dépasse 1 et
augmente (valeur ajoutée négative) sont seulement signalées.

Noms de secteurs : le fichier de matching sépare les secteurs par ';' et écrit
donc ',' à la place des ';' des noms EXIOBASE (ex. 'Wearing apparel, furs (18)'
pour 'Wearing apparel; furs (18)') ; les noms sont comparés après remplacement
des ';' par ',' et réduction des espaces multiples (sector_key).
"""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import ArpackNoConvergence, eigs

import extract_S_SSPs as ssp
from compute_ds_x_dom import MATMAT_DIR, MODULE_PLANEFR_DIR, get_ssp_base_accounts_dir, make_pickle_portable

MATCHING_FILE = MODULE_PLANEFR_DIR / "data" / "SSP" / "Matching_extensions_EXIOBASE_IMAGE.xlsx"
BRIDGE_FILE = MATMAT_DIR / "data" / "4-PlaneFR" / "Settings" / "4-engine" / "bridge_SSPs.xlsx"
# Comptes Monde choqués par le moteur (path_in.accounts de settings_engine_SSPs.json) :
# seule source de la structure et de l'année de base des chocs (voir docstring)
BASE_ACCOUNTS_DIR = get_ssp_base_accounts_dir()
BASE_EXTENSIONS_DIR = BASE_ACCOUNTS_DIR / "extensions"
BASE_SYSTEM_DIR = BASE_ACCOUNTS_DIR / "system"
SHOCKS_OUTPUT_DIR = MATMAT_DIR / "data" / "4-PlaneFR" / "2-shocks" / "3-SSPs_2050"
PROJ_YEAR = "2050"

NON_SCENARIO_SHEETS = {"Matching", "Régions", "Notes"}
S_RATIO_LABEL = "S 2050/2020"

# Marqueur, dans la colonne 'Domaine', d'un choc sur A (et non sur S)
A_SHOCK_MARKER = "→"
# Groupes d'utilisateurs (étapes 1 et 2) : domaines commençant par ce préfixe
# et comptant plusieurs lignes (une par vecteur énergétique)
ENERGY_GROUP_PREFIX = "Énergie → "
# Mix électrique (étape 3) : une ligne par filière
ELECTRICITY_MIX_DOMAIN = "Mix électrique → tous utilisateurs"
# Groupes dont seule l'intensité est choquée (parts EXIOBASE des vecteurs
# conservées) ; vider pour leur appliquer aussi les parts IMAGE (sensibilité)
HEAVY_GROUPS = {"Énergie → Sidérurgie", "Énergie → Chimie", "Énergie → Minéraux non métalliques"}
# Rayon spectral maximal admis pour A' (voir check_shocked_a ; 0,72 pour A 2015)
MAX_SPECTRAL_RADIUS = 0.95
# Part (en euros) en dessous de laquelle un vecteur est considéré absent de la
# colonne (coefficients résiduels, souvent importés, qui recevraient sinon toute
# une part IMAGE : ex. biomasse US dans les colonnes CA, x50 sur CH4 combustion)
MIN_VECTOR_SHARE = 1e-2
DA_FILE = "dA.pkl"

# Libellés du fichier de matching dont les segments ne se retrouvent pas tels
# quels dans les indicateurs EXIOBASE : segments à rechercher à la place.
ROW_ALIASES = {
    "Water Consumption Blue - Electricity - Electricity by hydro": [
        "Water Consumption Blue",
        "Electricity hydropower",
    ],
}

# Libellés dont les indicateurs choqués doivent en plus contenir l'un des
# secteurs de 'Secteurs EXIOBASE', ex. 'Cropland' + 'Cattle' ->
# 'Cropland - fallowed area-Cattle'.
SECTOR_IN_ROW_NAME = {"Cropland"}


def get_scenario_sheets():
    return [name for name in pd.ExcelFile(MATCHING_FILE).sheet_names if name not in NON_SCENARIO_SHEETS]


def load_region_bridge():
    """Région EXIOBASE -> région IMAGE agrégée (feuille 'regions' de bridge_SSPs.xlsx)."""
    bridge = pd.read_excel(BRIDGE_FILE, sheet_name="regions", header=[0, 1], index_col=[0, 1])
    bridge = bridge.droplevel(0, axis=0).droplevel(0, axis=1)
    return {exio: bridge.columns[bridge.loc[exio] == 1][0] for exio in bridge.index}


def load_scenario_shocks(sheet_name):
    """Reconstruit un tableau propre : la ligne 1 porte les noms de champ et les
    régions IMAGE (une région toutes les 3 colonnes), la ligne 2 les
    sous-colonnes (F, x, S 2050/2020) ; les données commencent ligne 3.
    Renvoie un DataFrame avec les colonnes domain, row_label, sectors puis une
    colonne par région IMAGE contenant S 2050/2020."""
    raw = pd.read_excel(MATCHING_FILE, sheet_name=sheet_name, header=None)
    regions = raw.iloc[0].ffill()
    sub_labels = raw.iloc[1]
    data = raw.iloc[2:].reset_index(drop=True)

    shocks = pd.DataFrame(
        {"domain": data[0].str.strip(), "row_label": data[1].str.strip(), "sectors": data[2]}
    )
    for col in raw.columns[3:]:
        if sub_labels[col] == S_RATIO_LABEL:
            shocks[regions[col]] = pd.to_numeric(data[col], errors="coerce")
    return shocks.dropna(subset=["domain", "row_label"])


def match_indicators(row_label, indicators):
    """Indicateurs de S_x_dom correspondant à un libellé 'Extension EXIOBASE'."""
    names = [ind[-1] if isinstance(ind, tuple) else ind for ind in indicators]
    if row_label in names:
        return [indicators[names.index(row_label)]]
    # Index à plusieurs niveaux, ex. ('ghg_combustion', 'CO2') <-> 'ghg_combustion - CO2'
    full_names = [" - ".join(map(str, ind)) if isinstance(ind, tuple) else ind for ind in indicators]
    if row_label in full_names:
        return [indicators[full_names.index(row_label)]]
    parts = ROW_ALIASES.get(row_label, [p.strip() for p in row_label.split(" - ")])
    parts = [p.lower() for p in parts]
    return [ind for ind, name in zip(indicators, names) if all(p in name.lower() for p in parts)]


def compute_ds(shocks, s_base, region_bridge):
    ds = pd.DataFrame(0.0, index=s_base.index, columns=s_base.columns)
    exio_regions = ds.columns.get_level_values("region")
    exio_sectors = sector_keys(ds.columns.get_level_values("sector"))
    image_regions = exio_regions.map(region_bridge)

    rules = []
    for _, row in shocks.iterrows():
        label = row["row_label"]
        sectors = [] if pd.isna(row["sectors"]) else [s.strip() for s in str(row["sectors"]).split(";") if s.strip()]

        matched = match_indicators(label, list(ds.index))
        if label in SECTOR_IN_ROW_NAME and sectors:
            matched = [ind for ind in matched if any(s.lower() in ind[-1].lower() for s in sectors)]
        if not matched:
            print(f"[skip] '{label}' ({row['sectors']}) : aucun indicateur correspondant dans S_x_dom, ignoré.")
            continue

        if not sectors:
            sector_mask = pd.Series(True, index=ds.columns)
        else:
            unknown = sorted(set(sector_keys(sectors)) - set(exio_sectors))
            if unknown:
                print(f"[warn] '{label}' : secteurs inconnus ignorés : {unknown}")
            sector_mask = pd.Series(exio_sectors.isin(sector_keys(sectors)), index=ds.columns)

        # dS par colonne (region, sector), à partir de la région IMAGE de rattachement
        ds_cols = pd.Series(image_regions.map(lambda r: row.get(r)), index=ds.columns).astype(float) - 1
        ds_cols = ds_cols[sector_mask.values & ds_cols.notna().values]
        rules.append((len(matched), label, matched, ds_cols))

    # Les règles les plus génériques d'abord : les plus spécifiques écrasent ensuite
    for _, label, matched, ds_cols in sorted(rules, key=lambda r: -r[0]):
        n_cells = 0
        for ind in matched:
            # Seulement les secteurs EXIOBASE non nuls (dans au moins une région)
            # pour cet indicateur dans les comptes de base
            s_row = s_base.loc[ind]
            active_sectors = set(s_row[s_row != 0].index.get_level_values("sector"))
            cols = ds_cols[ds_cols.index.get_level_values("sector").isin(active_sectors)]
            ds.loc[ind, cols.index] = cols.values
            n_cells += len(cols)
        print(f"[shock] '{label}' -> {len(matched)} indicateur(s), {n_cells} cellule(s)")

    return ds


def split_sectors(value):
    """Liste de secteurs EXIOBASE séparés par ';' (liste vide si case vide)."""
    return [] if pd.isna(value) else [s.strip() for s in str(value).split(";") if s.strip()]


def sector_key(name):
    """Nom de secteur comparable entre EXIOBASE et le fichier de matching (voir docstring)."""
    return re.sub(r"\s+", " ", str(name).replace(";", ",")).strip()


def sector_keys(names):
    return pd.Index([sector_key(name) for name in names])


def load_base_year():
    """base_year des comptes de base (system/info.json), reporté dans les info.json
    des chocs."""
    info = json.loads((BASE_SYSTEM_DIR / "info.json").read_text(encoding="utf-8-sig"))
    return str(info["base_year"])


def load_engine_a():
    """A des comptes de base auxquels le moteur applique le choc (BASE_ACCOUNTS_DIR)."""
    path = BASE_SYSTEM_DIR / "A.pkl"
    print(f"[A] A de base : {path.resolve()}")
    return pd.read_pickle(path)


def write_info(info_path, **fields):
    """Écrit (ou met à jour) un info.json de choc : les champs donnés remplacent
    ceux d'un fichier existant, dont les autres champs sont conservés. Un
    base_year hérité d'une exécution sur d'autres comptes de base est ainsi
    corrigé."""
    info = json.loads(info_path.read_text(encoding="utf-8-sig")) if info_path.exists() else {}
    info.update(fields)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def make_a_context(a_base, region_bridge):
    """Repères de lignes/colonnes de A utilisés par les étapes du choc sur A."""
    col_regions = np.asarray(a_base.columns.get_level_values("region"))
    return SimpleNamespace(
        columns=a_base.columns,
        row_keys=sector_keys(a_base.index.get_level_values("sector")),
        col_keys=sector_keys(a_base.columns.get_level_values("sector")),
        col_regions=col_regions,
        col_image_regions=np.asarray([region_bridge.get(r) for r in col_regions]),
    )


def select_rows(ctx, products):
    return np.flatnonzero(ctx.row_keys.isin(sector_keys(products)))


def select_cols(ctx, sectors):
    return np.flatnonzero(ctx.col_keys.isin(sector_keys(sectors)))


def warn_unknown_sectors(domain, products, sectors, ctx):
    for kind, names, known in (("lignes", products, ctx.row_keys), ("colonnes", sectors, ctx.col_keys)):
        unknown = sorted(set(sector_keys(names)) - set(known))
        if unknown:
            print(f"[warn] '{domain}' : secteurs ({kind}) inconnus ignorés : {unknown}")


def load_a_rules():
    """Lignes de chocs sur A de la feuille Matching, par domaine : produits (lignes de A),
    secteurs (colonnes) et formules IMAGE F et x."""
    matching = ssp.load_matching().dropna(subset=["Extension EXIOBASE"])
    rules = {}
    for _, row in matching.iterrows():
        domain = str(row["Domaine"]).strip()
        if A_SHOCK_MARKER in domain:
            rules.setdefault(domain, []).append({
                "products": split_sectors(row["Extension EXIOBASE"]),
                "sectors": split_sectors(row["Secteurs EXIOBASE"]),
                "f": row["Numérateur F"],
                "x": row["Dénominateur x"],
            })
    return rules


def a_rule_kind(domain, n_rows):
    """'mix' (étape 3), 'group' (étapes 1 et 2) ou 'factor' (facteur unique)."""
    if domain == ELECTRICITY_MIX_DOMAIN:
        return "mix"
    if domain.startswith(ENERGY_GROUP_PREFIX) and n_rows > 1:
        return "group"
    return "factor"


def load_image_data(sheet_name, image_map, baseline):
    """Données IMAGE du scénario (csv), valeurs 2020 communes (scénario de référence)."""
    csv_by_sheet = {path.stem[:31]: path for path in ssp.SCENARIOS_DIR.glob("*.csv")}
    return ssp.with_common_baseline(ssp.load_scenario(csv_by_sheet[sheet_name], image_map), baseline)


def image_values(cell, data, region):
    """Valeurs IMAGE [2020, 2050] de la formule d'une cellule de Matching ; None si
    la cellule n'a pas de formule ou si la valeur n'est pas calculable."""
    tree, _ = ssp.parse_formula(cell)
    if tree is None:
        return None
    values = ssp.evaluate(tree, [data], region)
    return values if np.isfinite(values).all() else None


def share_deltas(values):
    """Δ = part 2050 - part 2020 de chaque vecteur dans la somme des vecteurs ayant
    une valeur IMAGE (Δ = 0 pour les autres, ou si la somme est nulle)."""
    total = sum((v for v in values if v is not None), np.zeros(2))
    if total.min() <= 0:
        return np.zeros(len(values))
    return np.array([0.0 if v is None else v[1] / total[1] - v[0] / total[0] for v in values])


def group_targets(rows, data, regions):
    """Par région IMAGE : (I, Δ par vecteur) d'un groupe d'utilisateurs (voir docstring)."""
    def levels(region):
        f_values = [image_values(row["f"], data, region) for row in rows]
        fe = sum((v for v in f_values if v is not None), np.zeros(2))
        act = next((v for v in (image_values(row["x"], data, region) for row in rows) if v is not None), None)
        with np.errstate(divide="ignore", invalid="ignore"):
            intensity = fe / act if act is not None else np.full(2, np.nan)
        return f_values, intensity

    _, world_intensity = levels(ssp.WORLD_REGION)
    targets = {}
    for region in regions:
        f_values, intensity = levels(region)
        i = intensity[1] / intensity[0] if intensity[0] > 0 else np.nan
        if not np.isfinite(i) or i < 0 or intensity[0] < ssp.MIN_RELATIVE_INTENSITY * world_intensity[0]:
            i = 1.0
        targets[region] = (i, share_deltas(f_values))
    return targets


def reshape_vectors(a_new, ctx, vector_rows, cols_all, targets, apply_shares):
    """Étapes 1-2 (groupe) ou 3 (mix, I = 1) sur les colonnes cols_all : total des
    vecteurs multiplié par I, puis parts en euros décalées de Δ si apply_shares, sur
    les seuls vecteurs présents dans la colonne (voir docstring). Chaque vecteur est
    mis à l'échelle au prorata de ses coefficients.
    Renvoie le nombre de couples (vecteur absent, colonne) dont Δ > 0 n'a pas pu être
    appliqué faute de coefficient à choquer."""
    n_absent = 0
    for region, (i, delta) in targets.items():
        cols = cols_all[ctx.col_image_regions[cols_all] == region]
        if not len(cols):
            continue
        totals = np.array([a_new[np.ix_(rows, cols)].sum(axis=0) for rows in vector_rows])
        if apply_shares:
            energy = totals.sum(axis=0)
            shares = np.divide(totals, energy, out=np.zeros_like(totals), where=energy > 0)
            present = shares >= MIN_VECTOR_SHARE
            n_absent += int((~present & (energy > 0) & (delta[:, None] > 0)).sum())
            new_shares = np.where(present, np.maximum(0.0, shares + delta[:, None]), 0.0)
            norm = new_shares.sum(axis=0)
            new_shares = np.where(norm > 0, new_shares / np.where(norm > 0, norm, 1.0), shares)
            new_totals = i * energy * new_shares
        else:
            new_totals = i * totals
        for rows, vector_new_totals, vector_totals in zip(vector_rows, new_totals, totals):
            scale = np.divide(vector_new_totals, vector_totals, out=np.ones_like(vector_totals), where=vector_totals > 0)
            a_new[np.ix_(rows, cols)] *= scale
    return n_absent


def apply_factor_rules(a_new, ctx, factor_shocks):
    """Étape 1 du docstring : A' = A x S 2050/2020 (feuille du scénario)."""
    for _, row in factor_shocks.iterrows():
        products, sectors = split_sectors(row["row_label"]), split_sectors(row["sectors"])
        warn_unknown_sectors(row["domain"], products, sectors, ctx)
        rows, cols_all = select_rows(ctx, products), select_cols(ctx, sectors)
        factors = np.asarray([row.get(r, np.nan) for r in ctx.col_image_regions[cols_all]], dtype=float)
        shocked = np.isfinite(factors) & (factors != 1)
        cols, factors = cols_all[shocked], factors[shocked]
        if len(rows) and len(cols):
            a_new[np.ix_(rows, cols)] *= factors
        print(f"[shock A] '{row['domain']}' (facteur) -> {len(rows)} ligne(s) x {len(cols)} colonne(s)")


def compute_shocked_a(a_base, ctx, a_shocks, rules, data):
    """A' = A choqué (étapes 1 à 3 du docstring), en tableau numpy."""
    a_new = a_base.copy()
    regions = sorted(r for r in set(ctx.col_image_regions) if r is not None)
    kinds = {domain: a_rule_kind(domain, len(rows)) for domain, rows in rules.items()}

    apply_factor_rules(a_new, ctx, a_shocks[a_shocks["domain"].map(kinds) == "factor"])

    for domain, rows in rules.items():
        if kinds[domain] != "group":
            continue
        sectors = sorted({s for row in rows for s in row["sectors"]})
        warn_unknown_sectors(domain, [p for row in rows for p in row["products"]], sectors, ctx)
        targets = group_targets(rows, data, regions)
        n_absent = reshape_vectors(
            a_new, ctx,
            vector_rows=[select_rows(ctx, row["products"]) for row in rows],
            cols_all=select_cols(ctx, sectors),
            targets=targets,
            apply_shares=domain not in HEAVY_GROUPS,
        )
        print(f"[shock A] '{domain}' ({'intensité' if domain in HEAVY_GROUPS else 'intensité + parts'}"
              f", {n_absent} vecteur(s) x colonne(s) absent(s) non créé(s)) : I = "
              + ", ".join(f"{r} {i:.2f}" for r, (i, _) in targets.items()))

    for domain, rows in rules.items():
        if kinds[domain] != "mix":
            continue
        sectors = sorted({s for row in rows for s in row["sectors"]})
        warn_unknown_sectors(domain, [p for row in rows for p in row["products"]], sectors, ctx)
        deltas = {r: share_deltas([image_values(row["f"], data, r) for row in rows]) for r in regions}
        n_absent = reshape_vectors(
            a_new, ctx,
            vector_rows=[select_rows(ctx, row["products"]) for row in rows],
            cols_all=select_cols(ctx, sectors),
            targets={r: (1.0, delta) for r, delta in deltas.items()},
            apply_shares=True,
        )
        print(f"[shock A] '{domain}' : {n_absent} filière(s) x colonne(s) absente(s) non créée(s)")

    return a_new


def spectral_radius(a):
    """Rayon spectral de A (plus grande valeur propre en module), np.inf si non convergé."""
    try:
        return float(abs(eigs(csr_matrix(a), k=1, which="LM", return_eigenvectors=False, maxiter=5000)[0]))
    except ArpackNoConvergence:
        return np.inf


def check_shocked_a(a_new, a_base, ctx, sheet_name):
    """Renvoie True si le rayon spectral de A' est < MAX_SPECTRAL_RADIUS (inverse de
    Leontief fini et positif). Signale aussi les colonnes dont la somme dépasse 1 et
    augmente (valeur ajoutée négative)."""
    col_sums, base_sums = a_new.sum(axis=0), a_base.sum(axis=0)
    above = np.flatnonzero((col_sums >= 1) & (col_sums > base_sums + 1e-9))
    if len(above):
        print(f"[warn A] {sheet_name} : {len(above)} colonne(s) de A' de somme >= 1 en hausse "
              f"(valeur ajoutée négative) :")
        for j in above[np.argsort(-col_sums[above])][:15]:
            print(f"    {ctx.columns[j]} : {base_sums[j]:.3f} -> {col_sums[j]:.3f}")

    rho, rho_base = spectral_radius(a_new), spectral_radius(a_base)
    if rho < MAX_SPECTRAL_RADIUS:
        print(f"[check A] {sheet_name} : rayon spectral de A' = {rho:.4f} (A de base : {rho_base:.4f})")
        return True
    print(f"[ERREUR A] {sheet_name} : rayon spectral de A' = {rho:.4f} >= {MAX_SPECTRAL_RADIUS:g} "
          f"(A de base : {rho_base:.4f}) : inverse de Leontief non fiable, choc sur A non écrit.")
    return False


def relative_shock(a_new, a_base):
    """dA = A'/A - 1 là où A > 0, 0 ailleurs (A' = A x (1 + dA) dans le moteur), en place
    dans a_new pour limiter la mémoire."""
    np.divide(a_new, a_base, out=a_new, where=a_base > 0)
    a_new -= 1.0
    a_new[a_base <= 0] = 0.0
    return a_new


def remove_da_output(sheet_name):
    """Supprime un dA.pkl d'une exécution précédente, pour ne pas le réutiliser par erreur."""
    path = SHOCKS_OUTPUT_DIR / f"{sheet_name}_{PROJ_YEAR}" / "system" / DA_FILE
    if path.exists():
        path.unlink()
        print(f"[remove] {path}")


def write_da_outputs(sheet_name, da):
    out_dir = SHOCKS_OUTPUT_DIR / f"{sheet_name}_{PROJ_YEAR}" / "system"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_pickle_portable(da).to_pickle(out_dir / DA_FILE)
    print(f"[write] {out_dir / DA_FILE}")


def write_ds_outputs(sheet_name, ext_name, ds, base_year):
    out_dir = SHOCKS_OUTPUT_DIR / f"{sheet_name}_{PROJ_YEAR}" / "extensions" / ext_name
    out_dir.mkdir(parents=True, exist_ok=True)
    if not isinstance(ds.index, pd.MultiIndex):
        ds.index = pd.MultiIndex.from_arrays([ds.index], names=ds.index.names)
    make_pickle_portable(ds).to_pickle(out_dir / "dS_x_dom.pkl")

    # detail_levels identique à celui de l'extension dans les comptes de base
    shutil.copy2(BASE_EXTENSIONS_DIR / ext_name / "detail_levels.xlsx", out_dir / "detail_levels.xlsx")

    write_info(
        out_dir / "info.json",
        base_year=base_year,
        proj_year=PROJ_YEAR,
        scenario_name=sheet_name,
        extension_name=ext_name,
    )
    print(f"[write] {out_dir / 'dS_x_dom.pkl'} (+ detail_levels.xlsx, info.json)")


def write_system_outputs(sheet_name, base_year):
    """detail_levels.xlsx de system identique à celui des comptes de base."""
    out_dir = SHOCKS_OUTPUT_DIR / f"{sheet_name}_{PROJ_YEAR}" / "system"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(BASE_SYSTEM_DIR / "detail_levels.xlsx", out_dir / "detail_levels.xlsx")

    write_info(out_dir / "info.json", base_year=base_year, proj_year=PROJ_YEAR, scenario_name=sheet_name)
    print(f"[write] {out_dir / 'detail_levels.xlsx'} (+ info.json)")


def main():
    region_bridge = load_region_bridge()
    base_year = load_base_year()
    print(f"[base] comptes de base : {BASE_ACCOUNTS_DIR.resolve()} (base_year {base_year})")
    a_base = None

    for sheet_name in get_scenario_sheets():
        write_system_outputs(sheet_name, base_year)
        shocks = load_scenario_shocks(sheet_name)

        is_a_shock = shocks["domain"].str.contains(A_SHOCK_MARKER, regex=False)
        a_shocks, shocks = shocks[is_a_shock], shocks[~is_a_shock]
        if not a_shocks.empty:
            print(f"=== {sheet_name} / A ===")
            if a_base is None:
                # Chargés une fois pour tous les scénarios
                a_base_df = load_engine_a()
                a_base, ctx = a_base_df.to_numpy(), make_a_context(a_base_df, region_bridge)
                rules = load_a_rules()
                image_map, _, _ = ssp.load_region_maps()
                baseline = ssp.load_scenario(ssp.SCENARIOS_DIR / f"{ssp.BASELINE_SCENARIO}.csv", image_map)
            data = load_image_data(sheet_name, image_map, baseline)
            a_new = compute_shocked_a(a_base, ctx, a_shocks, rules, data)
            if check_shocked_a(a_new, a_base, ctx, sheet_name):
                da = relative_shock(a_new, a_base)
                write_da_outputs(sheet_name, pd.DataFrame(da, index=a_base_df.index, columns=a_base_df.columns))
            else:
                remove_da_output(sheet_name)
            del a_new

        for domain, domain_shocks in shocks.groupby("domain", sort=False):
            ext_name = str(domain).strip().lower()
            print(f"=== {sheet_name} / {ext_name} ===")
            s_base = pd.read_pickle(BASE_EXTENSIONS_DIR / ext_name / "S_x_dom.pkl")
            ds = compute_ds(domain_shocks, s_base, region_bridge)
            write_ds_outputs(sheet_name, ext_name, ds, base_year)


if __name__ == "__main__":
    main()
