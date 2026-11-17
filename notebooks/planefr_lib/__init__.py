"""
planefr_lib: library shared by the PlaneFR analysis notebooks.

Gathers code that used to be duplicated across several notebooks (data
loading, filtering/weighting by characterization factor, "overshoot / safe
operating space" figures, bubble charts, stacked bars).

Usage from a notebook in the notebooks/ folder (see the existing notebooks for
the full snippet, which locates this folder dynamically instead of using a
hard-coded path):

    import sys
    from pathlib import Path

    notebooks_dir = Path.cwd()
    while not (notebooks_dir / "planefr_lib").exists() and notebooks_dir != notebooks_dir.parent:
        notebooks_dir = notebooks_dir.parent
    if str(notebooks_dir) not in sys.path:
        sys.path.insert(0, str(notebooks_dir))

    from planefr_lib import config, io, processing, colors
    from planefr_lib.plot_overshoot import create_overshoot_safe_space_figure_by_region
"""
