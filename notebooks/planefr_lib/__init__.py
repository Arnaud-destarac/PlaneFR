"""
planefr_lib : bibliothèque partagée par les notebooks d'analyse de PlaneFR.

Regroupe le code auparavant dupliqué dans plusieurs notebooks (chargement des
données, filtrage/pondération par facteur de caractérisation, figures
"overshoot / safe operating space", bubble charts, barres empilées).

Utilisation depuis un notebook du dossier notebooks/ :

    import sys
    sys.path.insert(0, r"C:\\Users\\Arnaud\\Documents\\PlaneFR\\notebooks")
    from planefr_lib import config, io, processing, colors
    from planefr_lib.plot_overshoot import create_overshoot_safe_space_figure
"""
