"""
Helpers de couleur : catégories de consommation (barres empilées, bubble charts)
et style par scénario (figures overshoot).
"""

import re

from . import config


def sort_categories(categories):
    """Trie une liste de catégories selon config.CATEGORY_ORDER.

    Les catégories inconnues sont conservées à la fin, dans leur ordre d'origine.
    """
    known = [c for c in config.CATEGORY_ORDER if c in categories]
    unknown = [c for c in categories if c not in config.CATEGORY_ORDER]
    return known + unknown


def get_category_color(category, fallback_idx=0):
    """Couleur par nom de catégorie (stable, indépendante de la position après agrégation)."""
    if category in config.CATEGORY_COLOR_MAP:
        return config.CATEGORY_COLOR_MAP[category]
    fallback_colors = list(config.CATEGORY_COLORS.values())
    return fallback_colors[fallback_idx % len(fallback_colors)]


def get_dark_shade(hex_color, factor=0.7):
    """Retourne une nuance foncée d'une couleur hex."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    dark_rgb = tuple(int(c * factor) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*dark_rgb)


def get_light_shade(hex_color, factor=1.3):
    """Retourne une nuance claire d'une couleur hex."""
    h = hex_color.lstrip("#")
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    light_rgb = tuple(min(255, int(c * factor)) for c in rgb)
    return "#{:02x}{:02x}{:02x}".format(*light_rgb)


def scenario_style_map(scenario_names):
    """Associe à chaque nom de dossier de scénario une couleur de bulle {face, edge}.

    Reconnaît (motifs cherchés dans le nom en minuscules) :
      - "2015"          -> blanc (base-year)
      - "world"          -> vert
      - "europe"         -> rouge
      - "2019"           -> bleu clair
      - "trend"          -> gris très foncé (tendanciel 2050)
      - "suff"/"s2"/"s3"/"tech" -> dégradé de bleus (scénarios de transition 2050)
      - sinon            -> gris-bleu générique (fallback)

    Cette liste couvre les noms de dossiers actuels (Base_year_2015,
    Sufficiency_NZE_2050, Tech_NZE_2050, TREND_2050, 2019_France/World/Europe_27) —
    c'est la version qui doit rester la référence si de nouveaux motifs de noms de
    scénarios apparaissent (voir notebooks/README ou le plan de refactor pour le
    détail : l'ancien mapping "s1".."s4"/"tend" utilisé dans une version antérieure
    du code ne correspondait déjà plus aux noms de scénarios actuels).
    """
    styles = {}
    blue_scale = {"suff": "#dbdbdb", "s2": "#9ec9f8", "s3": "#4f90d8", "tech": "#6f6f6f"}

    for raw_name in scenario_names:
        name_low = raw_name.lower()
        if "world" in name_low:
            styles[raw_name] = {"face": "#1f791f", "edge": "#1f791f"}
        elif "europe" in name_low:
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
