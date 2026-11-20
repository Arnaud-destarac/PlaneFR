"""
100 % stacked bar showing the contribution of each Earth-system process (LP,
"EXIOBASE groups" column of indicator_computing.xlsx) to the footprint of
a given indicator (e.g. "Biodiversity loss"), one bar per scenario.

Unlike plot_stacked_bar (stacked by consumption category, one indicator per
bar), this figure stacks by LP of origin, for a single indicator, with one
bar per scenario.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from . import figure_export

# Grouping of the raw LPs into display categories: ghg_combustion and
# ghg_emissions are two separate extractions of the same physical phenomenon
# (GHG emissions from combustion vs. other emissions) and are therefore
# merged into a single "Climate Change" segment.
PROCESS_GROUP_MAP = {
    "ghg_combustion": "climate_change",
    "ghg_emissions": "climate_change",
    "land": "land",
    "water": "water",
    "nutrients": "nutrients",
    "air_emissions": "air_emissions",
    "raw_materials": "raw_materials",
}

PROCESS_GROUP_LABELS = {
    "climate_change": "GHG emissions",
    "land": "Land use",
    "water": "Water Consumption",
    "nutrients": "N & P surpluses",
    "air_emissions": "Other air emissions",
    "raw_materials": "Raw materials",
}

# Stacking order (bottom -> top). An LP missing from PROCESS_GROUP_MAP is
# grouped under "air_emissions"/"Other" -- see group_lp_values.
PROCESS_GROUP_ORDER = ["climate_change", "land", "water", "nutrients", "air_emissions", "raw_materials"]

PROCESS_GROUP_COLORS = {
    "climate_change": "#F57C00",
    "land": "#1e5631",
    "water": "#5b9bd5",
    "nutrients": "#c8a878",
    "air_emissions": "#4e17d1",
    "raw_materials": "#9e9e9e",
}

# A process group appears in the legend only if its share reaches this
# threshold (%) in at least one scenario.
LEGEND_MIN_SHARE_PCT = 0.5


def group_lp_values(lp_values):
    """Group {raw_lp: value} into {group: value} via PROCESS_GROUP_MAP
    (unknown LP -> "air_emissions"/"Other", catch-all)."""
    grouped = {}
    for lp_name, value in lp_values.items():
        group = PROCESS_GROUP_MAP.get(lp_name, "air_emissions")
        grouped[group] = grouped.get(group, 0.0) + value
    return grouped


def create_biodiversity_process_contribution(lp_values_by_scenario, scenario_names,
                                             indicator_name="", ax=None, show_legend=True,
                                             unit="", excel_path=None,
                                             legend_min_share=LEGEND_MIN_SHARE_PCT):
    """Stacked bar in % (one per scenario) of the contribution of each process
    group (PROCESS_GROUP_ORDER) to the footprint of indicator_name
    (biodiversity_process_contribution script).

    Args:
        lp_values_by_scenario: list of {raw_lp: absolute_value}, one entry per
            scenario (see processing.process_indicator_lp_breakdown).
        scenario_names: matching scenario names (same order/length).
        indicator_name: used for the figure title and the Excel workbook.
        ax: existing matplotlib axis; if None, a new figure is created.
        unit: unit of the absolute values (Excel workbook only).
        excel_path: if given, Excel workbook of the plotted data (see
            figure_export): "Contribution" sheet (value and share of each
            process group, by scenario) and "LP detail" (raw LPs before grouping).
        legend_min_share: only process groups whose share (%) reaches this
            value in at least one scenario are shown in the legend.

    Returns:
        (fig, ax) if ax=None was given as input, otherwise (None, ax).
    """
    grouped_by_scenario = [group_lp_values(lp_values) for lp_values in lp_values_by_scenario]

    if excel_path is not None:
        contribution_rows, lp_rows = [], []
        for scenario_name, grouped, lp_values in zip(scenario_names, grouped_by_scenario, lp_values_by_scenario):
            total = sum(grouped.values())
            for group in PROCESS_GROUP_ORDER:
                value = grouped.get(group, 0.0)
                share = figure_export.ratio(value, total)
                contribution_rows.append({
                    "Scenario": scenario_name, "Indicator": indicator_name, "Unit": unit,
                    "Process group": PROCESS_GROUP_LABELS[group],
                    "Value": figure_export.clean_value(value),
                    "Share of the footprint (%)": 100 * share if share is not None else None,
                })
            for lp_name, value in lp_values.items():
                lp_rows.append({
                    "Scenario": scenario_name, "Indicator": indicator_name, "Unit": unit,
                    "LP (EXIOBASE group)": lp_name,
                    "Process group": PROCESS_GROUP_LABELS[PROCESS_GROUP_MAP.get(lp_name, "air_emissions")],
                    "Value": figure_export.clean_value(value),
                })
        figure_export.write_figure_excel(excel_path, {"Contribution": contribution_rows, "LP detail": lp_rows},
                                         about={
            "Figure": f"Contribution by pressure categories — {indicator_name}",
            "Generated by": "planefr_lib.plot_process_breakdown.create_biodiversity_process_contribution",
            "Content": "Consumption-based footprint of the indicator broken down by Earth-system process "
                       "(LP of origin), by scenario; bars = share of each process group.",
            "Unit": unit or "\"Figures unit\" of boundaries_values.xlsx",
        })
    n_scenarios = len(scenario_names)

    if ax is None:
        fig, ax = plt.subplots(figsize=(1.3 * max(n_scenarios, 3) + 3, 7))
    else:
        fig = None

    x_pos = np.arange(n_scenarios)
    bar_width = 0.6
    totals = np.array([sum(g.values()) for g in grouped_by_scenario])

    bottom = np.zeros(n_scenarios)
    legend_groups = []
    for group in PROCESS_GROUP_ORDER:
        heights = np.array([g.get(group, 0.0) for g in grouped_by_scenario])
        heights_pct = np.where(totals > 0, heights / totals * 100, 0)
        if np.max(np.abs(heights_pct), initial=0) >= legend_min_share:
            legend_groups.append(group)
        ax.bar(x_pos, heights_pct, bar_width, bottom=bottom,
               label=PROCESS_GROUP_LABELS[group], color=PROCESS_GROUP_COLORS[group],
               edgecolor="white", linewidth=0.5)
        for idx, (h, b) in enumerate(zip(heights_pct, bottom)):
            if h >= 3:
                ax.text(idx, b + h / 2, f"{h:.0f}%", ha="center", va="center",
                        fontsize=9.5, fontweight="bold", color="white")
        bottom = bottom + heights_pct

    ax.set_ylabel("Share of footprint (%)", fontsize=12, fontweight="bold")
    title = f"Contribution by pressure categories — {indicator_name}" if indicator_name else "Contribution by impact categories"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(scenario_names, rotation=30, ha="right", fontweight="bold", fontsize=10.5)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    if show_legend:
        # Top of the stack first, so the legend reads in the same order as the bars
        handles = [Patch(facecolor=PROCESS_GROUP_COLORS[g], edgecolor="white") for g in reversed(legend_groups)]
        labels = [PROCESS_GROUP_LABELS[g] for g in reversed(legend_groups)]
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11)

    if fig is not None:
        plt.tight_layout()

    return (fig, ax) if fig is not None else (None, ax)
