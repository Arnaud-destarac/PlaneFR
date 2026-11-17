"""
planefr_lib : bibliothèque partagée par les notebooks d'analyse de PlaneFR.

Regroupe le code auparavant dupliqué dans plusieurs notebooks (chargement des
données, filtrage/pondération par facteur de caractérisation, figures
"overshoot / safe operating space", bubble charts, barres empilées).

Utilisation depuis un notebook du dossier notebooks/ (voir les notebooks
existants pour le snippet complet, qui localise ce dossier dynamiquement au
lieu d'un chemin codé en dur) :

    import sys
    from pathlib import Path

    notebooks_dir = Path.cwd()
    while not (notebooks_dir / "planefr_lib").exists() and notebooks_dir != notebooks_dir.parent:
        notebooks_dir = notebooks_dir.parent
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    from planefr_lib import config, io, processing, colors
    from planefr_lib.plot_overshoot import create_overshoot_safe_space_figure
"""
