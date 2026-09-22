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

    Reconnaît, dans l'ordre :
      - code région "W" (Monde)   -> vert
      - code région "EU27"        -> rouge
    puis (motifs cherchés dans le nom en minuscules) :
      - "2015"           -> blanc (base-year)
      - "2019"           -> bleu clair
      - "trend"          -> gris très foncé (tendanciel 2050)
      - "suff"/"s2"/"s3"/"tech" -> dégradé de bleus (scénarios de transition 2050)
      - sinon            -> gris-bleu générique (fallback)

    Les deux codes région sont cherchés sur des morceaux entiers du nom (découpé
    sur tout ce qui n'est pas alphanumérique) et non par inclusion de texte, pour
    que "W" ne corresponde qu'à "2019_W" et pas à n'importe quel nom contenant
    un w. Ils remplacent les motifs "world"/"europe" de la version précédente,
    devenus caducs avec le renommage des dossiers (2019_World -> 2019_W,
    2019_Europe_27 -> 2019_EU27) — même convention de codes que
    processing.resolve_region_code.

    Cette liste couvre les noms de dossiers actuels (Base_year, Sufficiency_NZE,
    Tech_NZE, TREND, 2019_FR/W/EU27) — c'est la version qui doit rester la
    référence si de nouveaux motifs de noms de scénarios apparaissent (voir
    notebooks/README ou le plan de refactor pour le détail : l'ancien mapping
    "s1".."s4"/"tend" utilisé dans une version antérieure du code ne
    correspondait déjà plus aux noms de scénarios actuels).
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


# Couleurs par code région EXIOBASE (figure par région,
# plot_overshoot.create_overshoot_safe_space_figure_by_region) : "FR" reprend la
# couleur "2019" de scenario_style_map, "EU27"/"W" leurs couleurs dédiées ci-dessus.
REGION_CODE_COLORS = {
    "W": {"face": "#1f791f", "edge": "#1f791f"},
    "EU27": {"face": "#c90808", "edge": "#c90808"},
    "FR": {"face": "#54a6f7", "edge": "#54a6f7"},
}
REGION_CODE_FALLBACK = {"face": "#b7c4d3", "edge": "#44505c"}


def region_code_style_map(scenario_codes):
    """Associe à chaque nom de scénario une couleur de bulle {face, edge}, à
    partir de son code région EXIOBASE -- et non de son nom, contrairement à
    scenario_style_map : deux scénarios différents rattachés au même code (ex.
    Base_year et TREND, tous deux "FR" par défaut, cf.
    processing.resolve_region_code) reçoivent donc la même couleur.

    "FR" -> bleu clair, "EU27" -> rouge, "W" -> vert (REGION_CODE_COLORS) ; tout
    autre code -> gris générique (REGION_CODE_FALLBACK).

    Args:
        scenario_codes: {nom_scénario: code EXIOBASE}, tel que retourné par
            processing.resolve_region_code pour chaque scénario.
    """
    return {
        name: REGION_CODE_COLORS.get(str(code).upper(), REGION_CODE_FALLBACK)
        for name, code in scenario_codes.items()
    }
