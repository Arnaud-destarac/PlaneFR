"""
Color helpers: consumption categories (stacked bars, bubble charts) and
per-scenario style (overshoot figures).
"""

import re

from . import config


def sort_categories(categories):
    """Sort a list of categories according to config.CATEGORY_ORDER.

    Unknown categories are kept at the end, in their original order.
    """
    known = [c for c in config.CATEGORY_ORDER if c in categories]
    unknown = [c for c in categories if c not in config.CATEGORY_ORDER]
    return known + unknown


def get_category_color(category, fallback_idx=0):
    """Color by category name (stable, independent of the position after aggregation)."""
    if category in config.CATEGORY_COLOR_MAP:
        return config.CATEGORY_COLOR_MAP[category]
    fallback_colors = list(config.CATEGORY_COLORS.values())
    return fallback_colors[fallback_idx % len(fallback_colors)]


def get_dark_shade(hex_color, factor=0.7):
    """Return a darker shade of a hex color."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    dark_rgb = tuple(int(c * factor) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*dark_rgb)


def get_light_shade(hex_color, factor=1.3):
    """Return a lighter shade of a hex color."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    light_rgb = tuple(min(255, int(c * factor)) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*light_rgb)


def scenario_style_map(scenario_names):
    """Map each scenario folder name to a bubble color {face, edge}.

    Recognizes, in this order:
      - region code "W" (World)   -> green
      - region code "EU27"        -> red
    then (patterns searched in the lower-case name):
      - "2015"           -> white (base year)
      - "2019"           -> light blue
      - "trend"          -> very dark grey (2050 trend scenario)
      - "suff"/"s2"/"s3"/"tech" -> shades of blue (2050 transition scenarios)
      - otherwise        -> generic grey-blue (fallback)

    The two region codes are matched against whole tokens of the name (split
    on every non-alphanumeric character) rather than by substring, so that "W"
    only matches "2019_W" and not any name containing a w. They replace the
    "world"/"europe" patterns of the previous version, which became obsolete
    when the folders were renamed (2019_World -> 2019_W, 2019_Europe_27 ->
    2019_EU27) -- same code convention as processing.resolve_region_code.

    This list covers the current folder names (Base_year, Sufficiency_NZE,
    Tech_NZE, TREND, 2019_FR/W/EU27) -- it is the version that should remain
    the reference if new scenario name patterns appear (the old
    "s1".."s4"/"tend" mapping used in an earlier version of the code already
    no longer matched the current scenario names).
    """
    styles = {}
    blue_scale = {"suff": "#dbdbdb", "s2": "#9ec9f8", "s3": "#4f90d8", "tech": "#6f6f6f"}

    for raw_name in scenario_names:
        name_low = raw_name.lower()
        tokens = {t.upper() for t in re.split(r"[^A-Za-z0-9]+", raw_name) if t}
        if "W" in tokens:
            styles[raw_name] = {"face": "#1f791f", "edge": "#1f791f"}
        elif "EU27" in tokens:
            styles[raw_name] = {"face": "#c90808", "edge": "#c90808"}
        elif "2015" in name_low:
            styles[raw_name] = {"face": "white", "edge": "#1f1f1f"}
        elif "2019" in name_low:
            styles[raw_name] = {"face": "#54a6f7", "edge": "#54a6f7"}
        elif "trend" in name_low:
            styles[raw_name] = {"face": "#212121", "edge": "#212121"}
        else:
            matched = False
            for s_key, blue_color in blue_scale.items():
                if re.search(rf"(^|[^a-z0-9]){s_key}([^a-z0-9]|$)", name_low):
                    styles[raw_name] = {"face": blue_color, "edge": "#1f3a5a"}
                    matched = True
                    break
            if not matched:
                styles[raw_name] = {"face": "#b7c4d3", "edge": "#44505c"}

    return styles


# Colors by EXIOBASE region code (by-region figure,
# plot_overshoot.create_multi_regions_synthesis): "FR" reuses the
# "2019" color of scenario_style_map, "EU27"/"W" their dedicated colors above.
REGION_CODE_COLORS = {
    "W": {"face": "#1f791f", "edge": "#1f791f"},
    "EU27": {"face": "#c90808", "edge": "#c90808"},
    "FR": {"face": "#54a6f7", "edge": "#54a6f7"},
}
REGION_CODE_FALLBACK = {"face": "#b7c4d3", "edge": "#44505c"}


def region_code_style_map(scenario_codes):
    """Map each scenario name to a bubble color {face, edge}, based on its
    EXIOBASE region code -- not on its name, unlike scenario_style_map: two
    different scenarios attached to the same code (e.g. Base_year and TREND,
    both "FR" by default, see processing.resolve_region_code) therefore get
    the same color.

    "FR" -> light blue, "EU27" -> red, "W" -> green (REGION_CODE_COLORS); any
    other code -> generic grey (REGION_CODE_FALLBACK).

    Args:
        scenario_codes: {scenario_name: EXIOBASE code}, as returned by
            processing.resolve_region_code for each scenario.
    """
    return {
        name: REGION_CODE_COLORS.get(str(code).upper(), REGION_CODE_FALLBACK)
        for name, code in scenario_codes.items()
    }
