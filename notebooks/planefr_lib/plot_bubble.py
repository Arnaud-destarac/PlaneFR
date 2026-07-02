"""
Bubble chart par sous-processus : pour chaque catégorie de consommation (Food,
Housing, Mobility, Final goods, Final services), 3 bulles superposées
verticalement (M_imp, M_row, M_dom), taille proportionnelle à la demande finale
associée (Y_dom pour M_imp/M_dom, Y_imp pour M_row).

Agrégation par MOYENNE PONDÉRÉE : pour chaque catégorie, chaque produit marqué
dans le bridge 'M' contribue à M_categorie pondéré par le Y de la catégorie
correspondante (Housing -> Y_Residential, Mobility -> Y_Transport, sinon
correspondance positionnelle) — voir build_bridge_to_y_mapping().

Deux figures :
  - plot_bubble_chart            : 1 scénario (référence), 1 subplot par sous-processus.
  - plot_bubble_chart_comparison : 2 scénarios (ex. base-year vs un scénario 2050),
                                    axe Y indépendant par scénario dans chaque subplot.
"""

import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import colors, io, processing

# Mise à l'échelle des tailles de bulles (valeurs de la version "production" du
# code, cf. plan de refactor — pas les 200.0 d'un brouillon antérieur superseded).
BUBBLE_SCALE = 1600.0
BUBBLE_MIN_SIZE = 30.0

BUBBLE_COLORS = {
    "M_imp": "#1f77b4",
    "M_row": "#ff7f0e",
    "M_dom": "#2ca02c",
}


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


def compute_subprocess_m_vectors(scenario_folder_path, facteurs_carac_df, bridge_m, y_dom_5, y_imp_5):
    """M_dom/M_imp/M_row agrégés par sous-processus (somme sur tous les LP
    associés), réduits aux 5 catégories de consommation par moyenne pondérée.

    Returns:
        (data_by_subprocess: {nom: {"M_dom", "M_imp", "M_row"}}, y_dom_cat, y_imp_cat)
    """
    subprocess_to_lp = processing.get_unique_subprocesses(facteurs_carac_df)
    bridge_to_y_col = build_bridge_to_y_mapping(list(bridge_m.columns), list(y_dom_5.columns))

    data_by_subprocess = {}
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

        data_by_subprocess[subprocess_name] = {
            "M_dom": weighted_mean_by_category(agg_dom, y_dom_5, bridge_m, bridge_to_y_col),
            "M_imp": weighted_mean_by_category(agg_imp, y_dom_5, bridge_m, bridge_to_y_col),
            "M_row": weighted_mean_by_category(agg_row, y_imp_5, bridge_m, bridge_to_y_col),
        }

    y_dom_cat = aggregate_y_for_sizes(y_dom_5, bridge_m, bridge_to_y_col)
    y_imp_cat = aggregate_y_for_sizes(y_imp_5, bridge_m, bridge_to_y_col)
    return data_by_subprocess, y_dom_cat, y_imp_cat


def prepare_bubble_dataset(scenario_folder_path, facteurs_carac_df, bridge_m):
    """Charge Y et calcule les données bubble chart pour un scénario donné."""
    y_dom_5, y_imp_5 = io.get_y_blocks(scenario_folder_path)
    return compute_subprocess_m_vectors(scenario_folder_path, facteurs_carac_df, bridge_m, y_dom_5, y_imp_5)


def scale_bubble_sizes(values, scale=BUBBLE_SCALE, min_size=BUBBLE_MIN_SIZE):
    """Normalise les tailles de bulles pour éviter les bulles géantes, en gardant
    la proportion relative au maximum."""
    arr = np.asarray(values, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > 0), arr, 0.0)
    if arr.max() <= 0:
        return np.full_like(arr, min_size, dtype=float)
    return min_size + (arr / arr.max()) * scale


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
# FIGURES
# ============================================================================


def plot_bubble_chart(data_by_subprocess, y_dom_cat, y_imp_cat):
    """1 subplot par sous-processus, 3 bulles par catégorie (M_imp/M_row/M_dom) —
    scénario de référence uniquement."""
    subprocesses = list(data_by_subprocess.keys())
    if not subprocesses:
        print("Aucune donnée à tracer.")
        return None, None

    categories = list(y_dom_cat.index)
    x_pos = np.arange(len(categories))

    n_cols = 3
    n_rows = int(math.ceil(len(subprocesses) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    size_dom = scale_bubble_sizes(y_dom_cat.reindex(categories).values)
    size_imp = scale_bubble_sizes(y_imp_cat.reindex(categories).values)

    for i, sp in enumerate(subprocesses):
        ax = axes[i]
        payload = data_by_subprocess[sp]
        m_imp = payload["M_imp"].reindex(categories).fillna(0).values
        m_row = payload["M_row"].reindex(categories).fillna(0).values
        m_dom = payload["M_dom"].reindex(categories).fillna(0).values

        ax.scatter(x_pos, m_imp, s=size_dom, c=BUBBLE_COLORS["M_imp"], alpha=0.6, edgecolors="none",
                   label="M_imp (taille: y_dom)")
        ax.scatter(x_pos, m_row, s=size_imp, c=BUBBLE_COLORS["M_row"], alpha=0.6, edgecolors="none",
                   label="M_row (taille: y_imp)")
        ax.scatter(x_pos, m_dom, s=size_dom, c=BUBBLE_COLORS["M_dom"], alpha=0.6, edgecolors="none",
                   label="M_dom (taille: y_dom)")

        ax.set_title(sp, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.25, linestyle="--")

    for i in range(len(subprocesses), len(axes)):
        axes[i].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BUBBLE_COLORS[key], markersize=10,
               alpha=0.7, label=f"{key} (taille: {'y_imp' if key == 'M_row' else 'y_dom'})")
        for key in ["M_imp", "M_row", "M_dom"]
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Bubble chart - scénario de référence (1 subplot par sous-processus)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    return fig, axes


def plot_bubble_chart_comparison(base_data, base_y_dom_cat, base_y_imp_cat,
                                  s1_data, s1_y_dom_cat, s1_y_imp_cat,
                                  base_label="base-year_2015", s1_label="S1_2050"):
    """Figure comparative entre deux scénarios : 1 subplot par sous-processus,
    axe vertical gauche pour base_label, axe vertical droit indépendant pour
    s1_label, bulles décalées horizontalement pour rester lisibles côte à côte.
    Si un sous-processus est absent de s1_data, ses bulles sont tracées à 0."""
    subprocesses = list(base_data.keys())
    if not subprocesses:
        print("Aucune donnée à tracer.")
        return None, None

    categories = list(dict.fromkeys(list(base_y_dom_cat.index) + list(s1_y_dom_cat.index)))
    x_pos = np.arange(len(categories))
    x_base = x_pos - 0.12
    x_s1 = x_pos + 0.12

    n_cols = 3
    n_rows = int(math.ceil(len(subprocesses) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.8 * n_cols, 5.1 * n_rows))
    axes = np.array(axes).reshape(-1)

    base_colors = BUBBLE_COLORS
    s1_colors = {
        "M_imp": colors.get_light_shade(base_colors["M_imp"], factor=1.25),
        "M_row": colors.get_light_shade(base_colors["M_row"], factor=1.18),
        "M_dom": colors.get_light_shade(base_colors["M_dom"], factor=1.20),
    }

    base_size_dom = scale_bubble_sizes(base_y_dom_cat.reindex(categories).fillna(0).values)
    base_size_imp = scale_bubble_sizes(base_y_imp_cat.reindex(categories).fillna(0).values)
    s1_size_dom = scale_bubble_sizes(s1_y_dom_cat.reindex(categories).fillna(0).values)
    s1_size_imp = scale_bubble_sizes(s1_y_imp_cat.reindex(categories).fillna(0).values)

    legend_handles, legend_labels = [], []

    for i, sp in enumerate(subprocesses):
        ax_left = axes[i]
        ax_right = ax_left.twinx()
        ax_left.set_facecolor("none")
        ax_right.set_facecolor("none")
        ax_left.patch.set_alpha(0)
        ax_right.patch.set_alpha(0)
        ax_left.set_zorder(2)
        ax_right.set_zorder(1)

        s1_payload = s1_data[sp] if sp in s1_data else {
            "M_imp": pd.Series(0.0, index=categories),
            "M_row": pd.Series(0.0, index=categories),
            "M_dom": pd.Series(0.0, index=categories),
        }
        base_payload = base_data[sp]

        base_m_imp = base_payload["M_imp"].reindex(categories).fillna(0).values
        base_m_row = base_payload["M_row"].reindex(categories).fillna(0).values
        base_m_dom = base_payload["M_dom"].reindex(categories).fillna(0).values
        s1_m_imp = s1_payload["M_imp"].reindex(categories).fillna(0).values
        s1_m_row = s1_payload["M_row"].reindex(categories).fillna(0).values
        s1_m_dom = s1_payload["M_dom"].reindex(categories).fillna(0).values

        ax_left.scatter(x_base, base_m_imp, s=base_size_dom, c=base_colors["M_imp"], alpha=0.72,
                         marker="o", label=f"{base_label} - M_imp", edgecolors="none")
        ax_left.scatter(x_base, base_m_row, s=base_size_imp, c=base_colors["M_row"], alpha=0.72,
                         marker="o", label=f"{base_label} - M_row", edgecolors="none")
        ax_left.scatter(x_base, base_m_dom, s=base_size_dom, c=base_colors["M_dom"], alpha=0.72,
                         marker="o", label=f"{base_label} - M_dom", edgecolors="none")

        ax_right.scatter(x_s1, s1_m_imp, s=s1_size_dom, c=s1_colors["M_imp"], alpha=0.55, marker="o",
                          label=f"{s1_label} - M_imp", edgecolors=base_colors["M_imp"], linewidths=1.0)
        ax_right.scatter(x_s1, s1_m_row, s=s1_size_imp, c=s1_colors["M_row"], alpha=0.55, marker="o",
                          label=f"{s1_label} - M_row", edgecolors=base_colors["M_row"], linewidths=1.0)
        ax_right.scatter(x_s1, s1_m_dom, s=s1_size_dom, c=s1_colors["M_dom"], alpha=0.55, marker="o",
                          label=f"{s1_label} - M_dom", edgecolors=base_colors["M_dom"], linewidths=1.0)

        base_ymin, base_ymax = _compute_axis_limits([base_m_imp, base_m_row, base_m_dom])
        s1_ymin, s1_ymax = _compute_axis_limits([s1_m_imp, s1_m_row, s1_m_dom])
        ax_left.set_ylim(base_ymin, base_ymax)
        ax_right.set_ylim(s1_ymin, s1_ymax)

        ax_left.set_title(f"{sp}\n{base_label} vs {s1_label}", fontsize=11, fontweight="bold", pad=8)
        ax_left.set_xticks(x_pos)
        ax_left.set_xticklabels(categories, rotation=45, ha="right", fontsize=8)
        ax_left.set_ylabel(base_label, fontsize=9, fontweight="bold", color=base_colors["M_imp"])
        ax_right.set_ylabel(s1_label, fontsize=9, fontweight="bold", color=base_colors["M_row"])
        ax_left.tick_params(axis="y", labelsize=8, colors=base_colors["M_imp"])
        ax_right.tick_params(axis="y", labelsize=8, colors=base_colors["M_row"])
        ax_left.tick_params(axis="x", labelsize=8)
        ax_left.grid(axis="y", alpha=0.22, linestyle="--")
        ax_left.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)
        ax_right.axhline(0, color="#888888", linewidth=0.8, alpha=0.5)

        ax_left.spines["top"].set_visible(False)
        ax_left.spines["right"].set_visible(False)
        ax_right.spines["top"].set_visible(False)
        ax_right.spines["left"].set_visible(False)
        ax_right.tick_params(axis="x", bottom=False, labelbottom=False)

        if i == 0:
            for key in ["M_imp", "M_row", "M_dom"]:
                legend_handles.append(Line2D([0], [0], marker="o", linestyle="",
                                              markerfacecolor=base_colors[key], markeredgecolor="none",
                                              markersize=9, label=f"{base_label} - {key}"))
            for key in ["M_imp", "M_row", "M_dom"]:
                legend_handles.append(Line2D([0], [0], marker="o", linestyle="",
                                              markerfacecolor=s1_colors[key], markeredgecolor=base_colors[key],
                                              markersize=9, label=f"{s1_label} - {key}"))
            legend_labels = [h.get_label() for h in legend_handles]

    for i in range(len(subprocesses), len(axes)):
        axes[i].set_visible(False)

    if legend_handles:
        fig.legend(handles=legend_handles, labels=legend_labels, loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.01), frameon=True, fancybox=True, shadow=True)

    fig.suptitle(f"Bubble chart pondéré - comparaison {base_label} vs {s1_label}",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    return fig, axes
