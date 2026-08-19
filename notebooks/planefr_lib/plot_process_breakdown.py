"""
Barre empilée à 100 % représentant la contribution de chaque processus du
système Terre (LP, colonne "Processus du système Terre" de facteurs de
caractérisation.xlsx) à l'empreinte d'un sous-processus donné (ex.
"Biodiversity loss"), une barre par scénario.

Contrairement à plot_stacked_bar (empilement par catégorie de consommation,
un sous-processus par barre), cette figure empile par LP d'origine, pour un
seul sous-processus, avec une barre par scénario.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Regroupement des LP bruts en catégories d'affichage : ghg_combustion et
# ghg_emissions sont deux extractions distinctes du même phénomène physique
# (émissions de GES par combustion vs. autres émissions) et sont donc
# fusionnées sous un seul segment "Climate Change".
PROCESS_GROUP_MAP = {
    "ghg_combustion": "climate_change",
    "ghg_emissions": "climate_change",
    "land_use": "land_use",
    "water": "water",
    "biogeochemical": "biogeochemical",
    "air_emissions": "air_emissions",
}

PROCESS_GROUP_LABELS = {
    "climate_change": "GHG emissions",
    "land_use": "Land use",
    "water": "Water Consumption",
    "biogeochemical": "N & P surpluses",
    "air_emissions": "Other air emissions",
}

# Ordre d'empilement (bas -> haut). Un LP absent de PROCESS_GROUP_MAP (ex.
# raw_materials, sans rapport avec Biodiversity loss) est regroupé sous
# "air_emissions"/"Other", en fin de pile -- voir group_lp_values.
PROCESS_GROUP_ORDER = ["climate_change", "land_use", "water", "biogeochemical", "air_emissions"]

PROCESS_GROUP_COLORS = {
    "climate_change": "#F57C00",
    "land_use": "#1e5631",
    "water": "#5b9bd5",
    "biogeochemical": "#c8a878",
    "air_emissions": "#4e17d1",
}


def group_lp_values(lp_values):
    """Regroupe {lp_brut: valeur} en {groupe: valeur} via PROCESS_GROUP_MAP
    (LP inconnu -> "air_emissions"/"Other", fourre-tout)."""
    grouped = {}
    for lp_name, value in lp_values.items():
        group = PROCESS_GROUP_MAP.get(lp_name, "air_emissions")
        grouped[group] = grouped.get(group, 0.0) + value
    return grouped


def create_subprocess_contribution_chart(lp_values_by_scenario, scenario_names,
                                          subprocess_name="", ax=None, show_legend=True):
    """Barre empilée en % (une par scénario) de la contribution de chaque
    groupe de processus (PROCESS_GROUP_ORDER) à l'empreinte de subprocess_name.

    Args:
        lp_values_by_scenario: liste de {lp_brut: valeur_absolue}, une entrée
            par scénario (voir processing.process_subprocess_lp_breakdown).
        scenario_names: noms de scénario correspondants (même ordre/longueur).
        subprocess_name: utilisé seulement pour le titre de la figure.
        ax: axe matplotlib existant ; si None, une nouvelle figure est créée.

    Returns:
        (fig, ax) si ax=None a été fourni en entrée, sinon (None, ax).
    """
    grouped_by_scenario = [group_lp_values(lp_values) for lp_values in lp_values_by_scenario]
    n_scenarios = len(scenario_names)

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.3 * max(n_scenarios, 3) + 3, 7))
    else:
        fig = None

    x_pos = np.arange(n_scenarios)
    bar_width = 0.6
    totals = np.array([sum(g.values()) for g in grouped_by_scenario])

    bottom = np.zeros(n_scenarios)
    for group in PROCESS_GROUP_ORDER:
        heights = np.array([g.get(group, 0.0) for g in grouped_by_scenario])
        heights_pct = np.where(totals > 0, heights / totals * 100, 0)
        ax.bar(x_pos, heights_pct, bar_width, bottom=bottom,
               label=PROCESS_GROUP_LABELS[group], color=PROCESS_GROUP_COLORS[group],
               edgecolor="white", linewidth=0.5)
        for idx, (h, b) in enumerate(zip(heights_pct, bottom)):
            if h >= 3:
                ax.text(idx, b + h / 2, f"{h:.0f}%", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color="white")
        bottom = bottom + heights_pct

    ax.set_ylabel("Share of footprint (%)", fontsize=12, fontweight="bold")
    title = f"Contribution by pressure categories — {subprocess_name}" if subprocess_name else "Contribution by impact categories"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if show_legend:
        handles = [Patch(facecolor=PROCESS_GROUP_COLORS[g], edgecolor="white") for g in PROCESS_GROUP_ORDER]
        labels = [PROCESS_GROUP_LABELS[g] for g in PROCESS_GROUP_ORDER]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11)

    if fig is not None:
        plt.tight_layout()

    return (fig, ax) if fig is not None else (None, ax)
