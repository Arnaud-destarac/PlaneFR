"""
Bubble chart par sous-processus : pour chaque catégorie de consommation (Food,
Housing, Mobility, Final goods, Final services) ou chaque secteur, 2 bulles
superposées verticalement — M_dom+M_imp (produits français, taille Y_dom) et
M_row (produits importés, taille Y_imp).

Agrégation par MOYENNE PONDÉRÉE : pour chaque catégorie, chaque produit marqué
dans le bridge 'M' contribue à M_categorie pondéré par le Y de la catégorie
correspondante (Housing -> Y_Residential, Mobility -> Y_Transport, sinon
correspondance positionnelle) — voir build_bridge_to_y_mapping().

Pipeline en deux temps :
  1. export_subprocess_tables_excel : 1 classeur par scénario, 1 feuille par
     sous-processus (colonnes = secteurs + catégories, lignes = M_imp/M_dom/
     M_row/Y_imp/Y_dom et produits M*Y), secteurs triés selon la ligne choisie
     en B1 (modifiable à la main dans Excel).
  2. plot_bubble_chart : bubble chart relu depuis ces classeurs, par catégories
     de consommation ou par n premiers secteurs (type="categories"/"products"),
     baseline seule ou comparée à un 2e scénario (scenarios=False/True).
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import colors, io, processing

# Mise à l'échelle des tailles de bulles (valeurs de la version "production" du
# code, cf. plan de refactor — pas les 200.0 d'un brouillon antérieur superseded).
BUBBLE_SCALE = 1600.0
BUBBLE_MIN_SIZE = 30.0

# Séries tracées : (nom, lignes sommées pour l'ordonnée, ligne donnant la taille).
# M_dom et M_imp sont fusionnés : tous deux s'appliquent à Y_dom (produits
# français), et pour les catégories leur somme reste une moyenne pondérée par
# Y_dom (somme des moyennes pondérées par les mêmes poids).
BUBBLE_SERIES = [
    ("M_dom+M_imp", ("M_dom", "M_imp"), "Y_dom"),
    ("M_row", ("M_row",), "Y_imp"),
]

BUBBLE_COLORS = {
    "M_dom+M_imp": "#2ca02c",
    "M_row": "#ff7f0e",
}

# Couleurs des axes verticaux en mode comparaison (gauche : baseline, droite : scénario).
COMPARISON_AXIS_COLORS = ("#1f77b4", "#ff7f0e")


def build_bridge_to_y_mapping(bridge_cols, y_cols):
    """Associe chaque colonne du bridge M à une colonne de Y pour la pondération.

    Règles spéciales : Housing -> Residential, Mobility -> Transport, Food -> Food.
    Sinon, correspondance positionnelle (i-ème colonne bridge -> i-ème colonne Y).
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
    """Aligne un vecteur (200 produits) sur l'index de bridge_m, de manière robuste."""
    vec = vector.copy()
    if len(vec) == len(bridge_m):
        vec.index = bridge_m.index
    else:
        vec = vec.reindex(bridge_m.index)
    return vec.fillna(0.0)


def weighted_mean_by_category(m_vector, y_5cols, bridge_m, bridge_to_y_col):
    """M agrégé par catégorie via moyenne pondérée par Y :
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
    """Y par catégorie (somme), pour dimensionner les bulles."""
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


def compute_subprocess_m_sector_vectors(scenario_folder_path, facteurs_carac_df):
    """M_dom/M_imp/M_row par secteur (200 produits), agrégés par sous-processus
    (somme sur tous les LP associés, pondérée par les facteurs de caractérisation).

    Returns:
        {nom: {"M_dom", "M_imp", "M_row"}} — Series indexées par secteur.
    """
    subprocess_to_lp = processing.get_unique_subprocesses(facteurs_carac_df)

    sector_vectors = {}
    for subprocess_name, lp_list in subprocess_to_lp.items():
        agg_dom = agg_imp = agg_row = None
        extension_factor_map = processing.build_extension_factor_map(facteurs_carac_df, subprocess_name)

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
# EXPORT EXCEL
# ============================================================================

_EXCEL_SHEET_FORBIDDEN_CHARS = r'[]:*?/\\'


def _excel_sheet_name(name, used_names):
    """Nom de feuille Excel valide (31 caractères max, sans []:*?/\\) et unique."""
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

# Libellés de la colonne A qui délimitent le bloc "données sources" d'une feuille :
# écrits par _write_sortable_sheet, relus par load_subprocess_tables_excel.
SOURCE_BLOCK_TITLE = "Données sources (ordre du bridge, non triées)"
KEY_ROW_LABEL = "Clé de tri"
RANK_ROW_LABEL = "Rang"  # rang des secteurs ; sert aussi à compter les secteurs à la relecture
CATEGORY_RANK_ROW_LABEL = "Rang (catégories)"


# Catégories de Y.pkl exclues de la demande finale des tableaux Excel : GFCF
# (l'investissement est déjà endogénéisé dans M_k) et Exports.
EXCLUDED_Y_CATEGORIES = ("GFCF", "Exports")

# Secteurs outliers exclus des tableaux Excel (colonnes secteurs et agrégats par
# catégorie) : intensités M_row aberrantes (rapport RoW/dom de 50 à ~30 000) qui
# dominent GES et PDF.
EXCLUDED_SECTORS = (
    "Private households with employed persons (95)",
    "Rubber and plastic products (25)",
)


def build_subprocess_tables(scenario_folder_path, facteurs_carac_df, bridge_m):
    """Tableau par sous-processus : 1 colonne par secteur (200 produits du bridge M)
    + 1 colonne par catégorie de consommation, 1 ligne par donnée
    (M_imp, M_dom, M_row, Y_imp, Y_dom, puis les produits M*Y et leurs sommes).

    Colonnes catégories : M agrégés par moyenne pondérée par Y (même calcul que le
    bubble chart), Y agrégés par somme. Les produits M*Y d'une catégorie valent
    donc exactement la somme des M_i*Y_i de ses secteurs.

    Sources : M_dom = dom_{lp}/M_k.pkl, M_imp = imp_{lp}/M_k.pkl, M_row =
    imp_{lp}/M_RoW.pkl, Y = system/Y.pkl sans GFCF ni Exports (Y_dom = 200
    premières lignes, Y_imp = 200 suivantes). On a ainsi, par secteur :
        M_dom*Y_dom             = dom_{lp}/d_cba_k
        M_imp*Y_dom+M_row*Y_imp = imp_{lp}/d_cba_k
    (relation vérifiée exactement sur les fichiers, aux exports près : d_cba_k
    les inclut, ces tableaux non).

    Les secteurs de EXCLUDED_SECTORS sont retirés des colonnes secteurs et des
    agrégats par catégorie (lignes mises à 0 dans le bridge, ce qui conserve
    l'alignement positionnel sur les 200 produits).

    Returns:
        {nom_sous_processus: pd.DataFrame}
    """
    kept_sectors = ~bridge_m.index.isin(EXCLUDED_SECTORS)
    bridge_m = bridge_m.mul(kept_sectors, axis=0)
    kept_columns = np.concatenate([kept_sectors, np.ones(len(bridge_m.columns), dtype=bool)])

    y_dom_5, y_imp_5 = io.get_y_blocks(scenario_folder_path, excluded_y_categories=EXCLUDED_Y_CATEGORIES)
    bridge_to_y_col = build_bridge_to_y_mapping(list(bridge_m.columns), list(y_dom_5.columns))
    sector_vectors = compute_subprocess_m_sector_vectors(scenario_folder_path, facteurs_carac_df)

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
    """Écrit une feuille dont les colonnes sont triées par formules Excel, par ordre
    décroissant de la ligne nommée en B1 (liste déroulante, modifiable à la main) :
    les secteurs entre eux, puis les catégories entre elles. Disposition :
      - ligne 1          : choix de la ligne de tri (B1) ;
      - bloc du haut     : tableau trié (formules) — secteurs triés, puis catégories
                           triées, + une dernière ligne donnant la part (%) de
                           l'indicateur de tri dans son groupe ;
      - bloc du bas      : données sources (valeurs, ordre du bridge), suivies des
                           lignes de calcul "Clé de tri", "Rang" (secteurs) et
                           "Rang (catégories)".
    Rang = RANK + COUNTIF pour départager les ex-aequo (secteurs à 0 notamment),
    afin que chaque position 1..n corresponde à exactement une colonne.
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

    ws["A1"] = "Trier secteurs et catégories par (ordre décroissant) :"
    ws["A1"].font = bold
    ws["B1"] = sort_row
    ws["B1"].font = Font(bold=True, color="C00000")
    validation = DataValidation(type="list", formula1=f"={label_range}", allow_blank=False)
    ws.add_data_validation(validation)
    validation.add("B1")

    # Données sources (valeurs brutes, ordre du bridge)
    ws.cell(src_title, 1, SOURCE_BLOCK_TITLE).font = bold
    ws.cell(src_header, 1, subprocess_name).font = bold
    for j, name in enumerate(table.columns, start=2):
        ws.cell(src_header, j, str(name)).font = bold
    for i, label in enumerate(labels):
        ws.cell(src_first + i, 1, label).font = bold
        for j, value in enumerate(table.iloc[i].values, start=2):
            value = float(value)
            ws.cell(src_first + i, j, value if np.isfinite(value) else 0.0)

    # Clé de tri : valeur de la ligne choisie en B1, pour chaque colonne
    ws.cell(key_row, 1, KEY_ROW_LABEL).font = bold
    for j in range(2, 2 + n_cols):
        c = get_column_letter(j)
        ws.cell(key_row, j, f"=INDEX({c}${src_first}:{c}${src_last},MATCH($B$1,{label_range},0))")

    ws.cell(sorted_header, 1, subprocess_name).font = bold
    for i, label in enumerate(labels):
        ws.cell(sorted_first + i, 1, label).font = bold
    ws.cell(share_row, 1, '="Part de "&$B$1&" (%)"').font = Font(bold=True, color="C00000")

    # Tri de chaque groupe (secteurs, puis catégories) indépendamment : rang dans le
    # groupe, colonnes du tableau trié réordonnées, part (%) dans le total du groupe.
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


def export_subprocess_tables_excel(scenario_folder_path, facteurs_carac_df, bridge_m, output_path,
                                   sort_row=TOTAL_FOOTPRINT_ROW):
    """Écrit un classeur Excel avec 1 feuille par sous-processus (voir
    build_subprocess_tables pour le contenu, _write_sortable_sheet pour la mise en page).

    Les secteurs, puis les catégories, sont triés par ordre décroissant de `sort_row` via des formules :
    changer la cellule B1 d'une feuille dans Excel (liste déroulante) re-trie cette
    feuille. Les noms de feuilles sont tronqués à 31 caractères (limite Excel) ; le
    nom complet du sous-processus est rappelé en tête de chaque tableau.

    Returns:
        Path du fichier écrit.
    """
    from openpyxl import Workbook

    tables = build_subprocess_tables(scenario_folder_path, facteurs_carac_df, bridge_m)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()
    wb.remove(wb.active)
    used_names = set()
    for subprocess_name, table in tables.items():
        if sort_row not in table.index:
            raise ValueError(f"sort_row inconnu : {sort_row!r} (attendu : {list(table.index)})")
        ws = wb.create_sheet(_excel_sheet_name(subprocess_name, used_names))
        _write_sortable_sheet(ws, subprocess_name, table, table.shape[1] - len(bridge_m.columns), sort_row)
    wb.calculation.fullCalcOnLoad = True
    wb.save(output_path)
    return output_path


def load_subprocess_tables_excel(excel_path):
    """Relit un classeur écrit par export_subprocess_tables_excel.

    Lit le bloc "données sources" (valeurs, pas de formules) et la ligne de tri
    choisie en B1 de chaque feuille — éventuellement modifiée à la main dans
    Excel. Ne dépend pas du recalcul des formules du tableau trié : le classeur
    peut être relu qu'il ait été ouvert/enregistré dans Excel ou non.

    Secteurs et catégories sont rendus dans deux DataFrames séparés : certains
    noms existent dans les deux (ex. "Cattle", "Dairy products" avec le bridge
    M_split), une sélection par nom sur un tableau commun les confondrait.

    Returns:
        {nom_sous_processus: {"sector_table": DataFrame (lignes = données,
            colonnes = secteurs, ordre du bridge), "category_table": DataFrame
            (colonnes = catégories), "sort_row": str, "sectors": [...],
            "categories": [...]}}
    """
    from openpyxl import load_workbook

    wb = load_workbook(excel_path, read_only=True)
    sheets = {}
    for ws in wb.worksheets:
        rows = [list(r) for r in ws.iter_rows(values_only=True)]
        first_col = [r[0] if r else None for r in rows]
        if SOURCE_BLOCK_TITLE not in first_col:
            continue
        header_idx = first_col.index(SOURCE_BLOCK_TITLE) + 1
        key_idx = first_col.index(KEY_ROW_LABEL, header_idx)
        rank_idx = first_col.index(RANK_ROW_LABEL, key_idx)

        header = rows[header_idx]
        subprocess_name = header[0]
        n_cols = max(j for j, v in enumerate(header) if v is not None)
        columns = [str(v) for v in header[1:n_cols + 1]]
        # Les secteurs sont les colonnes classées (ligne "Rang" remplie), les
        # catégories celles qui suivent.
        n_sectors = sum(v is not None for v in rows[rank_idx][1:n_cols + 1])

        data_rows = rows[header_idx + 1:key_idx]
        values = np.array([[float(v) if v is not None else 0.0 for v in r[1:n_cols + 1]] for r in data_rows])
        labels = [r[0] for r in data_rows]
        sort_row = rows[0][1]
        if sort_row not in labels:
            raise ValueError(f"[{ws.title}] ligne de tri B1 inconnue : {sort_row!r} (attendu : {labels})")
        sheets[subprocess_name] = {
            "sector_table": pd.DataFrame(values[:, :n_sectors], index=labels, columns=columns[:n_sectors]),
            "category_table": pd.DataFrame(values[:, n_sectors:], index=labels, columns=columns[n_sectors:]),
            "sort_row": sort_row,
            "sectors": columns[:n_sectors],
            "categories": columns[n_sectors:],
            # False pour un classeur écrit avant le tri des catégories par formules
            "categories_sorted": CATEGORY_RANK_ROW_LABEL in first_col,
        }
    wb.close()
    return sheets


BUBBLE_ROWS = ["M_imp", "M_dom", "M_row", "Y_imp", "Y_dom"]


def select_bubble_items(sheet, type="categories", n=10):
    """Colonnes à tracer pour une feuille relue par load_subprocess_tables_excel.

    type="categories" : toutes les catégories de consommation ;
    type="products"   : les n premiers secteurs.
    Dans les deux cas, colonnes par ordre décroissant de la ligne de tri de la
    feuille (B1), ex-aequo dans l'ordre du bridge (même règle que le RANK +
    COUNTIF du classeur).

    Returns:
        DataFrame lignes BUBBLE_ROWS x colonnes sélectionnées.
    """
    if type == "categories":
        table, limit = sheet["category_table"], None
    elif type == "products":
        table, limit = sheet["sector_table"], n
    else:
        raise ValueError(f"type doit valoir 'categories' ou 'products', reçu {type!r}")
    order = np.argsort(-table.loc[sheet["sort_row"]].to_numpy(dtype=float), kind="stable")
    return table.iloc[:, order[:limit]].loc[BUBBLE_ROWS]


def scale_bubble_sizes(values, scale=BUBBLE_SCALE, min_size=BUBBLE_MIN_SIZE, ref_max=None):
    """Normalise les tailles de bulles pour éviter les bulles géantes, en gardant
    la proportion relative au maximum (`ref_max`, par défaut le max de `values` —
    à fournir pour que plusieurs subplots partagent la même échelle)."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    ref = arr.max() if ref_max is None else ref_max
    if arr.size == 0 or ref <= 0:
        return np.full_like(arr, min_size, dtype=float)
    return min_size + (arr / ref) * scale


def _compute_axis_limits(series_list, default_max=1.0):
    """Limites Y avec marge simple et robuste pour un ensemble de séries."""
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
# FIGURES (à partir des classeurs Excel de export_subprocess_tables_excel)
# ============================================================================

PRODUCT_LABEL_MAX_CHARS = 30


def _short_label(label, max_chars=PRODUCT_LABEL_MAX_CHARS):
    """Tronque les noms de secteurs EXIOBASE (souvent très longs) pour l'axe X."""
    label = str(label)
    return label if len(label) <= max_chars else label[:max_chars - 1] + "…"


def _scenario_label(excel_path):
    """Nom de scénario déduit du nom de fichier (bubble_data_by_subprocess_<scénario>.xlsx)."""
    return Path(excel_path).stem.replace("bubble_data_by_subprocess_", "")


def _series_values(payload, components):
    """Ordonnée d'une série de BUBBLE_SERIES : somme des lignes `components`."""
    return payload.loc[list(components)].to_numpy(dtype=float).sum(axis=0)


def _shares(sheet, items, type):
    """Part (%) de chaque élément affiché dans l'empreinte totale du sous-processus
    (TOTAL_FOOTPRINT_ROW, total = somme sur tous les secteurs de la feuille)."""
    total = float(sheet["sector_table"].loc[TOTAL_FOOTPRINT_ROW].sum())
    if total == 0:
        return np.zeros(len(items))
    table = sheet["category_table"] if type == "categories" else sheet["sector_table"]
    values = table.loc[TOTAL_FOOTPRINT_ROW].reindex(items).fillna(0.0).to_numpy(dtype=float)
    return 100.0 * values / total


SHARE_BAR_COLOR = "#b0b0b0"
SHARE_AXIS_LABEL = "Part dans le total (%)"


def _style_share_axis(ax_bar, shares_list):
    """Axe gauche des barres de part (%), sous les bulles."""
    top = max((float(np.max(s)) for s in shares_list if len(s)), default=0.0)
    ax_bar.set_ylim(0, top * 1.15 if top > 0 else 1.0)
    ax_bar.set_ylabel(SHARE_AXIS_LABEL, fontsize=9, color="#555555")
    ax_bar.tick_params(axis="y", labelsize=8, colors="#555555")
    ax_bar.spines["top"].set_visible(False)


def _ref_max(payloads, row):
    """Max d'une ligne (Y_dom/Y_imp) sur tous les subplots : échelle de bulles commune."""
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
    return f"{subprocess_name}\n(tri : {sheet['sort_row']})"


def _figure_subject(type, n):
    return "catégories de consommation" if type == "categories" else f"{n} premiers secteurs par sous-processus"


def plot_bubble_chart(base_excel_path, type="categories", n=10, scenarios=False,
                      comparison_excel_path=None, base_label=None, s1_label=None):
    """Bubble chart à partir des classeurs Excel de export_subprocess_tables_excel :
    1 subplot par sous-processus, 2 bulles par élément de l'axe X (cf.
    BUBBLE_SERIES) : M_dom+M_imp (taille Y_dom) et M_row (taille Y_imp).

    Args:
        base_excel_path: classeur du scénario de référence (baseline).
        type: "categories" -> toutes les catégories de consommation ;
            "products" -> les `n` premiers secteurs de chaque sous-processus.
            Dans les deux cas, par ordre décroissant de la ligne de tri choisie
            en B1 de sa feuille (modifiable à la main dans Excel, cf.
            select_bubble_items).
        n: nombre de secteurs affichés si type="products" (ignoré sinon).
        scenarios: False -> baseline seule ; True -> comparaison baseline vs
            `comparison_excel_path` (axe vertical indépendant par scénario). Les
            colonnes affichées et leur ordre sont ceux du classement de la
            baseline, pour comparer les mêmes éléments dans les deux scénarios.
        base_label, s1_label: légendes ; par défaut déduites des noms de fichiers.

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
        raise ValueError("scenarios=True nécessite comparison_excel_path")
    s1_sheets = load_subprocess_tables_excel(comparison_excel_path)
    s1_label = s1_label or _scenario_label(comparison_excel_path)
    s1_payloads, s1_shares = {}, {}
    for sp, base_payload in base_payloads.items():
        items = list(base_payload.columns)
        if sp in s1_sheets:
            table_key = "category_table" if type == "categories" else "sector_table"
            s1_payloads[sp] = s1_sheets[sp][table_key].reindex(index=BUBBLE_ROWS, columns=items).fillna(0.0)
            s1_shares[sp] = _shares(s1_sheets[sp], items, type)
        else:  # sous-processus absent du scénario comparé : bulles et barres tracées à 0
            s1_payloads[sp] = pd.DataFrame(0.0, index=BUBBLE_ROWS, columns=items)
            s1_shares[sp] = np.zeros(len(items))
    return _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                            base_label, s1_label)


def _plot_single(payloads, shares, titles, type, n, label):
    """1 subplot par sous-processus, 1 scénario : barres de part (%) dans le total
    sur l'axe gauche, bulles sur l'axe droit. Échelle des bulles commune à
    tous les subplots (max de Y_dom, resp. Y_imp, sur les éléments affichés)."""
    subprocesses = list(payloads)
    if not subprocesses:
        print("Aucune donnée à tracer.")
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
        ax.set_ylabel("Intensité M (bulles)", fontsize=9)
        ax.spines["top"].set_visible(False)

        ax_bar.set_title(titles[sp], fontsize=11 if type == "categories" else 10, fontweight="bold", pad=8)
        _style_x_axis(ax_bar, x_pos, list(payload.columns), type)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for i in range(len(subprocesses), len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUBBLE_COLORS[key], markersize=10,
               alpha=0.7, label=f"{key} (taille: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [Patch(facecolor=SHARE_BAR_COLOR, alpha=0.5, label=f"{SHARE_AXIS_LABEL} (axe gauche)")]
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles), fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Bubble chart - {label} - {_figure_subject(type, n)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def _plot_comparison(base_payloads, s1_payloads, base_shares, s1_shares, titles, type, n,
                     base_label, s1_label):
    """Comparaison de deux scénarios : barres de part (%) dans le total de chaque
    scénario sur l'axe gauche ; bulles sur deux axes droits indépendants (base_label
    contre le cadre, s1_label décalé vers l'extérieur), décalées horizontalement.
    Échelle des bulles propre à chaque scénario, commune à tous les subplots."""
    subprocesses = list(base_payloads)
    if not subprocesses:
        print("Aucune donnée à tracer.")
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
               markersize=9, label=f"{base_label} - {key} (taille: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=s1_colors[key], markeredgecolor=base_colors[key],
               markersize=9, label=f"{s1_label} - {key} (taille: {size_row})")
        for key, _, size_row in BUBBLE_SERIES
    ] + [
        Patch(facecolor=base_bar_color, alpha=0.25, label=f"{base_label} - {SHARE_AXIS_LABEL} (axe gauche)"),
        Patch(facecolor=s1_bar_color, alpha=0.25, label=f"{s1_label} - {SHARE_AXIS_LABEL} (axe gauche)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.01), frameon=True, fancybox=True, shadow=True)
    fig.suptitle(f"Bubble chart - {base_label} vs {s1_label} - {_figure_subject(type, n)}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig, axes
