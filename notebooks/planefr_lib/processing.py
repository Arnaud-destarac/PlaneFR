"""
Traitement des données chargées par io.py : filtrage/pondération par facteur de
caractérisation, agrégation par sous-processus, calcul des empreintes en valeurs
absolues (process_scenario) et par habitant CBA/PBA (process_scenario_per_capita).
"""

import pandas as pd

from . import colors, config, io

# ============================================================================
# FACTEURS DE CARACTÉRISATION : sous-processus <-> LP, extension -> facteur
# ============================================================================


def get_unique_subprocesses(facteurs_carac_df):
    """Retourne {sous-processus: [LP1, LP2, ...]} depuis les facteurs de caractérisation."""
    subprocess_to_lp = {}
    for _, row in facteurs_carac_df.iterrows():
        sp = row["Sous-processus"]
        lp = row["Processus du système Terre"]
        subprocess_to_lp.setdefault(sp, [])
        if lp not in subprocess_to_lp[sp]:
            subprocess_to_lp[sp].append(lp)
    return subprocess_to_lp


def build_extension_factor_map(facteurs_carac_df, subprocess_name, lp_name=None):
    """Construit {extension_exiobase: facteur_de_caractérisation} pour un sous-processus.

    Si lp_name est fourni, restreint aussi au LP correspondant. Les deux modes
    sont réellement utilisés (pas juste par commodité) : process_scenario() garde
    toutes les extensions du sous-processus quel que soit le LP en cours (chaque
    fichier d_cba ne contient de toute façon que les extensions de son propre LP),
    tandis que process_scenario_per_capita() restreint explicitement au LP — les
    deux donnent le même résultat numérique, mais ne pas fusionner les deux modes
    sans vérifier : ce n'est pas qu'une question de style.
    """
    mask = facteurs_carac_df["Sous-processus"] == subprocess_name
    if lp_name is not None:
        mask &= facteurs_carac_df["Processus du système Terre"] == lp_name
    subset = facteurs_carac_df[mask]
    return dict(zip(subset["Extensions exiobase"].values, subset["Facteurs de caractérisation"].values))


# ============================================================================
# FILTRAGE + PONDÉRATION PAR FACTEUR DE CARACTÉRISATION
# ============================================================================
# Primitive commune à tous les calculs d'empreinte : ne garder que les lignes
# (extensions exiobase) présentes dans extension_factor_map, et les multiplier
# par leur facteur. filter_and_weight() fait le travail ; les deux wrappers
# n'appliquent qu'une réduction différente (vecteur par catégorie, ou scalaire).


def filter_and_weight(df, extension_factor_map):
    """Filtre les lignes de `df` dont l'index (1er niveau si MultiIndex) est une
    clé de `extension_factor_map`, et les multiplie par le facteur correspondant.

    Returns:
        pd.DataFrame: les lignes pondérées retenues (peut être vide).
    """
    weighted_rows = [
        row * extension_factor_map[idx[0] if isinstance(idx, tuple) else idx]
        for idx, row in df.iterrows()
        if (idx[0] if isinstance(idx, tuple) else idx) in extension_factor_map
    ]
    return pd.DataFrame(weighted_rows) if weighted_rows else pd.DataFrame()


def filter_and_weight_vector(df, extension_factor_map):
    """Comme filter_and_weight(), réduit en sommant les lignes -> une Series (1
    valeur par colonne de `df`, donc par catégorie/secteur/région selon le cas)."""
    weighted = filter_and_weight(df, extension_factor_map)
    if weighted.empty:
        return pd.Series(dtype=float)
    summed = weighted.sum(axis=0)
    if isinstance(summed.index, pd.MultiIndex):
        summed.index = summed.index.get_level_values(-1)
    return summed


def filter_and_weight_scalar(df, extension_factor_map):
    """Comme filter_and_weight(), réduit en sommant tout -> un seul total (float)."""
    weighted = filter_and_weight(df, extension_factor_map)
    return float(weighted.to_numpy().sum()) if not weighted.empty else 0.0


# ============================================================================
# LOOKUPS SÛRS DANS seuils_df (évite de répéter le même try/except partout)
# ============================================================================


def lookup_seuil(seuils_df, row_name, column_name, require_positive=False):
    """Récupère seuils_df.loc[row_name, column_name] en float, en toute sécurité.

    Args:
        require_positive: True pour les seuils LB/UB (doivent être strictement
            positifs pour être valides) ; False pour un facteur de conversion,
            qui doit juste être non-nul (mais pourrait légitimement être négatif).

    Returns:
        float, ou None si la ligne/colonne n'existe pas, si la valeur est NaN,
        ou si elle échoue au test require_positive.
    """
    try:
        if row_name not in seuils_df.index or column_name not in seuils_df.columns:
            return None
        value = seuils_df.loc[row_name, column_name]
        if pd.isna(value):
            return None
        value = float(value)
        if require_positive and value <= 0:
            return None
        if not require_positive and value == 0:
            return None
        return value
    except (KeyError, TypeError, ValueError):
        return None


def lookup_first_text(seuils_df, row_names, column_name):
    """Essaie plusieurs noms de ligne candidats dans l'ordre (utile car seuils.xlsx
    n'est pas toujours orthographié à l'identique), retourne le premier texte
    non-NaN trouvé, ou "" sinon."""
    if isinstance(row_names, str):
        row_names = [row_names]
    try:
        for row_name in row_names:
            if row_name in seuils_df.index and column_name in seuils_df.columns:
                val = seuils_df.loc[row_name, column_name]
                if val is not None and str(val).lower() != "nan":
                    return str(val)
    except (KeyError, TypeError):
        pass
    return ""


# ============================================================================
# SEUILS RECALCULÉS SELON UN PRINCIPE DE PARTAGE ("sharing_principle")
# ============================================================================
# Remplace la lecture statique de "Equality (Lower bound)"/"Equality"/"Equality
# (Upper bound)" (et leurs variantes par habitant) par un partage égal per capita
# du budget mondial ("Lower safe bound"/"Safe limit"/"Upper safe bound"), pour une
# période de référence choisie par sharing_principle (voir
# config.SHARING_PRINCIPLE_POPULATION_ROW).


def lookup_population(pop_df, population_row, column_name):
    """Population en PERSONNES pour une ligne (année, ou "Moyenne ...") de la feuille
    "Population" de seuils.xlsx, qui stocke des milliers d'habitants (cf. io.load_population_df)."""
    return float(pop_df.loc[population_row, column_name]) * 1000


def compute_sharing_seuil(seuils_df, pop_df, subprocess_name, sharing_principle, threshold_row_name):
    """Seuil LOWER/LB/UB, par capita ou national (France), recalculé selon un principe
    de partage égalitaire du budget mondial, en remplacement de la valeur statique que
    lookup_seuil aurait lue pour `threshold_row_name` (une des 6 constantes
    THRESHOLD_LOWER/LB/UB_ABS/PER_CAPITA de config).

    Formule (partage égal per capita) :
        seuil_p_hab = Budget_mondial(LOWER, LB ou UB) / Conversion_budget_p_hab / Population_Monde(réf)
        seuil_france = Budget_mondial(LOWER, LB ou UB) / Conversion_budget / Population_Monde(réf) * Population_France(réf)

    où Budget_mondial(LOWER)="Lower safe bound", Budget_mondial(LB)="Safe limit",
    Budget_mondial(UB)="Upper safe bound" (dans l'unité mondiale "Unit budget"),
    Conversion_budget/Conversion_budget_p_hab convertissent cette unité mondiale vers
    "Figures unit"/"Figures unit (p.cap)" (mêmes lignes utilisées en diviseur que
    CONVERSION_ROW_ABS/CONVERSION_ROW_PER_CAPITA), et réf est la ligne de la feuille
    "Population" associée à sharing_principle.

    Returns:
        float, ou None si une donnée nécessaire est absente (budget/conversion manquant
        pour ce sous-processus, population introuvable...).
    """
    if sharing_principle not in config.SHARING_PRINCIPLE_POPULATION_ROW:
        raise ValueError(f"sharing_principle inconnu : {sharing_principle!r}")
    if pop_df is None:
        raise ValueError("pop_df est requis quand sharing_principle est fourni")

    is_ub = threshold_row_name in (config.THRESHOLD_UB_ABS, config.THRESHOLD_UB_PER_CAPITA)
    is_lower = threshold_row_name in (config.THRESHOLD_LOWER_ABS, config.THRESHOLD_LOWER_PER_CAPITA)
    is_per_capita = threshold_row_name in (
        config.THRESHOLD_LOWER_PER_CAPITA, config.THRESHOLD_LB_PER_CAPITA, config.THRESHOLD_UB_PER_CAPITA,
    )

    if is_ub:
        world_budget_row = config.WORLD_BUDGET_ROW_UB
    elif is_lower:
        world_budget_row = config.WORLD_BUDGET_ROW_LOWER
    else:
        world_budget_row = config.WORLD_BUDGET_ROW_LB
    world_conversion_row = config.WORLD_CONVERSION_ROW_PER_CAPITA if is_per_capita else config.WORLD_CONVERSION_ROW_ABS

    world_budget = lookup_seuil(seuils_df, world_budget_row, subprocess_name, require_positive=True)
    world_conversion = lookup_seuil(seuils_df, world_conversion_row, subprocess_name)
    if world_budget is None or not world_conversion:
        return None

    population_row = config.SHARING_PRINCIPLE_POPULATION_ROW[sharing_principle]
    world_population = lookup_population(pop_df, population_row, "Monde")
    seuil = world_budget / world_conversion / world_population

    if not is_per_capita:
        seuil *= lookup_population(pop_df, population_row, "France")

    return seuil


def lookup_threshold(seuils_df, threshold_row_name, subprocess_name, pop_df=None, sharing_principle=None):
    """Point d'entrée unique pour un seuil LOWER/LB/UB dans les figures : lecture
    statique (comportement historique, lookup_seuil avec require_positive=True) si
    sharing_principle est None, sinon recalcul dynamique via compute_sharing_seuil."""
    if sharing_principle is None:
        return lookup_seuil(seuils_df, threshold_row_name, subprocess_name, require_positive=True)
    return compute_sharing_seuil(seuils_df, pop_df, subprocess_name, sharing_principle, threshold_row_name)


# ============================================================================
# EMPREINTE EN VALEURS ABSOLUES (Synthèse multi-scénarios, Overshoot multi-scénarios)
# ============================================================================


def process_single_subprocess_scenario(subprocess_name, lp_name, scenario_folder_path,
                                        facteurs_carac_df, bridge_matrices_df, seuils_df):
    """Empreinte (dom + imp, + F_Y le cas échéant) d'un sous-processus pour un LP
    et un scénario donnés, répartie par catégorie de consommation via la matrice
    bridge. Calcule aussi "pba" (production-based, F_x_dom + F_Y, une seule
    valeur non répartie par catégorie). Brique de base de process_scenario()."""
    d_cba_imp = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="imp")
    d_cba_dom = io.load_d_cba_k_france(scenario_folder_path, lp_name, origin="dom")
    f_y_tot_df = io.load_f_y_tot_france(scenario_folder_path, lp_name)
    f_x_dom_df = io.load_f_x_dom_france(scenario_folder_path, lp_name)

    if d_cba_imp is None or d_cba_dom is None:
        return None

    extension_factor_map = build_extension_factor_map(facteurs_carac_df, subprocess_name)

    vector_imp = filter_and_weight_vector(d_cba_imp, extension_factor_map)
    vector_dom = filter_and_weight_vector(d_cba_dom, extension_factor_map)

    if vector_imp.empty or vector_dom.empty:
        return None

    f_y_tot_value = None
    if f_y_tot_df is not None:
        f_y_tot_weighted = filter_and_weight_vector(f_y_tot_df, extension_factor_map)
        if not f_y_tot_weighted.empty:
            f_y_tot_value = float(f_y_tot_weighted.sum())

    # PBA (production-based accounting) : empreinte de production domestique
    # (F_x_dom) + F_Y, mêmes facteurs et conversion que le CBA ci-dessous,
    # mais sans répartition par catégorie ni par dom/imp (une seule valeur).
    pba_raw = filter_and_weight_scalar(f_x_dom_df, extension_factor_map) if f_x_dom_df is not None else 0.0
    if f_y_tot_value is not None:
        pba_raw += f_y_tot_value

    # La bridge répartit par catégorie de consommation ; elle ne s'applique
    # qu'aux scénarios France (World/Europe restent un total agrégé unique).
    scen_name = str(scenario_folder_path.name)
    if not (scen_name.endswith("Europe") or scen_name.endswith("World")):
        bridge_categories = bridge_matrices_df.iloc[1:, 6:]
        bridge_categories.index = bridge_matrices_df.iloc[1:, 5].values
        result_imp = bridge_categories.T @ vector_imp
        result_dom = bridge_categories.T @ vector_dom
    else:
        result_imp = vector_imp.copy()
        result_dom = vector_dom.copy()

    conversion_factor = lookup_seuil(seuils_df, config.CONVERSION_ROW_ABS, subprocess_name)
    if conversion_factor is not None:
        result_imp = result_imp / conversion_factor
        result_dom = result_dom / conversion_factor
    pba_value = pba_raw / conversion_factor if conversion_factor else pba_raw

    # F_Y : émissions directes de la demande finale, ajoutées comme catégorie à
    # part entière (non répartie par la bridge, déjà rattachée à la conso finale).
    if f_y_tot_value is not None:
        if conversion_factor:
            f_y_tot_value = f_y_tot_value / conversion_factor
        result_dom = result_dom.copy()
        result_imp = result_imp.copy()
        result_dom["F_Y"] = f_y_tot_value
        result_imp["F_Y"] = 0.0

    return {
        "domestique": result_dom,
        "importé": result_imp,
        "categories": result_imp.index.tolist(),
        "pba": pba_value,
    }


def process_scenario(scenario_folder_path, facteurs_carac_df, bridge_matrices_df, seuils_df):
    """Traite un scénario complet : agrège tous les LP d'un même sous-processus.

    Returns:
        tuple: (data_by_subprocess: {nom_sous_processus: résultat}, subprocess_to_lp)
    """
    subprocess_to_lp = get_unique_subprocesses(facteurs_carac_df)
    data_by_subprocess = {}

    for subprocess_name, lp_list in subprocess_to_lp.items():
        aggregated = None
        for lp_name in lp_list:
            result = process_single_subprocess_scenario(
                subprocess_name, lp_name, scenario_folder_path,
                facteurs_carac_df, bridge_matrices_df, seuils_df,
            )
            if result is None:
                continue
            if aggregated is None:
                aggregated = {
                    "domestique": result["domestique"].copy(),
                    "importé": result["importé"].copy(),
                    "categories": result["categories"],
                    "pba": result["pba"],
                }
            else:
                aggregated["domestique"] = aggregated["domestique"].add(result["domestique"], fill_value=0)
                aggregated["importé"] = aggregated["importé"].add(result["importé"], fill_value=0)
                aggregated["pba"] += result["pba"]

        if aggregated is not None:
            aggregated["categories"] = colors.sort_categories(aggregated["importé"].index.tolist())
            data_by_subprocess[subprocess_name] = aggregated

    return data_by_subprocess, subprocess_to_lp


def process_subprocess_lp_breakdown(subprocess_name, scenario_folder_path, facteurs_carac_df, seuils_df):
    """Empreinte absolue (dom + imp + F_Y) d'un sous-processus pour un scénario,
    décomposée par LP (processus du système Terre) d'origine -- contrairement à
    process_single_subprocess_scenario/process_scenario, qui répartissent par
    catégorie de consommation en sommant tous les LP ensemble, cette fonction
    garde chaque LP séparé et ne répartit pas par catégorie.

    Utilisé par plot_process_breakdown pour visualiser la part de chaque LP
    dans l'empreinte d'un sous-processus donné (ex. Biodiversity loss).
    Même logique de chargement dom/imp/World-Europe que process_scenario_per_capita.

    Returns:
        {lp_name: valeur_absolue} -- seulement les LP pour lesquels des données
        existent pour ce scénario (dict potentiellement vide, jamais de LP
        manquant explicitement mis à 0).
    """
    subprocess_to_lp = get_unique_subprocesses(facteurs_carac_df)
    lp_list = subprocess_to_lp.get(subprocess_name, [])

    is_world_europe = "World" in scenario_folder_path.name or "Europe" in scenario_folder_path.name
    conversion_factor = lookup_seuil(seuils_df, config.CONVERSION_ROW_ABS, subprocess_name)

    lp_values = {}
    for lp_name in lp_list:
        if is_world_europe and lp_name == "ghg_combustion":
            continue

        ext_map = build_extension_factor_map(facteurs_carac_df, subprocess_name, lp_name)
        total = 0.0
        has_data = False

        if is_world_europe:
            d_cba = io.load_d_cba_world_europe(scenario_folder_path, lp_name)
            if d_cba is not None:
                total += filter_and_weight_scalar(d_cba, ext_map)
                has_data = True
            f_y_tot = io.load_f_y_tot_world_europe(scenario_folder_path, lp_name)
            if f_y_tot is not None:
                total += filter_and_weight_scalar(f_y_tot, ext_map)
                has_data = True
        else:
            for origin in ("dom", "imp"):
                d_cba = io.load_d_cba_france(scenario_folder_path, lp_name, origin)
                if d_cba is not None:
                    total += filter_and_weight_scalar(d_cba, ext_map)
                    has_data = True
            f_y_tot = io.load_f_y_tot_france(scenario_folder_path, lp_name)
            if f_y_tot is not None:
                total += filter_and_weight_scalar(f_y_tot, ext_map)
                has_data = True

        if has_data:
            lp_values[lp_name] = total / conversion_factor if conversion_factor else total

    return lp_values


# ============================================================================
# EMPREINTE PAR HABITANT, CBA + PBA (Figure comparaison)
# ============================================================================


def get_population(scenario_path, pop_df):
    """Population en PERSONNES pour un scénario (pop_df stocke des milliers
    d'habitants). Année déduite du nom du dossier (2019/2015/sinon 2050),
    géographie déduite de "World"/"Europe"/sinon France."""
    path_str = str(scenario_path)
    year = 2019 if "2019" in path_str else (2015 if "2015" in path_str else 2050)
    if "World" in path_str:
        col = "Monde"
    elif "Europe" in path_str:
        col = "Europe"
    else:
        col = "France"
    return float(pop_df.loc[year, col]) * 1000


def process_scenario_per_capita(scenario_folder_path, facteurs_carac_df, seuils_df, pop_df):
    """Empreinte par habitant CBA (consumption-based, d_cba) et PBA
    (production-based, F_x_dom) pour tous les sous-processus d'un scénario.

    Formule : empreinte_par_hab = total_exiobase / Conversion(p.hab) / population.
    CBA additionne dom + imp (France) ou lit le fichier unique (World/Europe).
    PBA n'utilise que la part domestique. F_Y_tot.pkl, si présent pour un LP,
    est ajouté aux deux totaux. ghg_combustion est ignoré pour World/Europe
    (absent de ces données).

    Returns:
        {sous-processus: {"domestique": Series([cba_par_hab], index=["total"]),
                           "importé": Series([0.0], index=["total"]),
                           "pba": pba_par_hab (float), "categories": ["total"]}}
    """
    is_world_europe = "World" in scenario_folder_path.name or "Europe" in scenario_folder_path.name
    population = get_population(scenario_folder_path, pop_df)

    subprocess_to_lp = get_unique_subprocesses(facteurs_carac_df)
    data_by_subprocess = {}

    for subprocess_name, lp_list in subprocess_to_lp.items():
        total_cba = 0.0
        total_pba = 0.0
        has_data = False

        for lp_name in lp_list:
            if is_world_europe and lp_name == "ghg_combustion":
                continue

            ext_map = build_extension_factor_map(facteurs_carac_df, subprocess_name, lp_name)

            if is_world_europe:
                d_cba = io.load_d_cba_world_europe(scenario_folder_path, lp_name)
                if d_cba is not None:
                    total_cba += filter_and_weight_scalar(d_cba, ext_map)
                    has_data = True
                f_dom = io.load_f_x_dom_world_europe(scenario_folder_path, lp_name)
                if f_dom is not None:
                    total_pba += filter_and_weight_scalar(f_dom, ext_map)
                f_y_tot = io.load_f_y_tot_world_europe(scenario_folder_path, lp_name)
                if f_y_tot is not None:
                    f_y_tot_total = filter_and_weight_scalar(f_y_tot, ext_map)
                    total_cba += f_y_tot_total
                    total_pba += f_y_tot_total
                    has_data = True
            else:
                for origin in ("dom", "imp"):
                    d_cba = io.load_d_cba_france(scenario_folder_path, lp_name, origin)
                    if d_cba is not None:
                        total_cba += filter_and_weight_scalar(d_cba, ext_map)
                        has_data = True
                f_dom = io.load_f_x_dom_france(scenario_folder_path, lp_name)
                if f_dom is not None:
                    total_pba += filter_and_weight_scalar(f_dom, ext_map)
                f_y_tot = io.load_f_y_tot_france(scenario_folder_path, lp_name)
                if f_y_tot is not None:
                    f_y_tot_total = filter_and_weight_scalar(f_y_tot, ext_map)
                    total_cba += f_y_tot_total
                    total_pba += f_y_tot_total
                    has_data = True

        if not has_data:
            continue

        conversion = lookup_seuil(seuils_df, config.CONVERSION_ROW_PER_CAPITA, subprocess_name)
        if conversion is not None:
            per_capita_cba = total_cba / conversion / population
            per_capita_pba = total_pba / conversion / population
        else:
            per_capita_cba = total_cba / population
            per_capita_pba = total_pba / population

        data_by_subprocess[subprocess_name] = {
            "domestique": pd.Series([per_capita_cba], index=["total"]),
            "importé": pd.Series([0.0], index=["total"]),
            "pba": per_capita_pba,
            "categories": ["total"],
        }

    return data_by_subprocess
