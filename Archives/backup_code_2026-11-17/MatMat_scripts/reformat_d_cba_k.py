"""
Reformatage ponctuel des extensions France (sortie brute du pipeline CIRED/MatMat,
dossier INPUT) vers data/<version>/. Les fichiers sont recherchés dans chaque
dossier extensions/ (find_files), donc tous les scénarios sont traités, qu'ils soient directement sous
data/<version>/<scénario> ou imbriqués par SSP (data/<version>/<SSP>/<scénario>,
ex. no_SSP/S1_2050, SSP1_2050/TEND_2050). Chaque étape est idempotente : un
fichier déjà reformaté est réécrit à l'identique. Traitements :

- ghg_emissions : retire la part ghg_combustion (traitée à part ci-dessous),
  renomme le niveau d'index "gas" en "indicator".
- ghg_combustion : effondre en une seule ligne "CO2" ; imp_ghg_combustion n'existe
  pas en amont, on le crée à zéro (les émissions de combustion sont par
  construction 100% domestiques).
- raw_materials : conserve toutes les lignes (indexées par "indicator", au
  niveau "sector") et ajoute une ligne "RMC" = somme de toutes les matières
  premières.
- F_x_dom de ghg_combustion et raw_materials : ces deux extensions n'ont pas de
  F_x_dom.pkl fourni en amont ; on le recalcule à partir de F_Y.pkl + F_Z.pkl
  (part domestique uniquement), effondré de la même façon.
- M, M_k (et M_RoW côté imp) : lignes reformatées comme celles de d_cba.
- S_x_dom (ghg_emissions), S_Y / S_Z (ghg_combustion, raw_materials) : lignes
  reformatées comme celles de F_x_dom (part domestique uniquement).
"""

import os
import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd

# Version Exiobase (variable d'environnement PLANEFR_EXIOBASE_VERSION, positionnée
# par run_full_pipeline.ps1 / run_pipeline_from_engine.ps1)
EXIOBASE_VERSION = os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")
INPUT = Path(r"C:\Users\Arnaud\Documents\CIRED\PlaneFR\Code\Module_PlaneFR\data") / EXIOBASE_VERSION
OUTPUT = INPUT

# Dossiers extensions/ de tous les scénarios, listés une seule fois : un rglob sur
# tout INPUT par motif de fichier reparcourait l'arborescence complète à chaque
# appel (très lent).
EXTENSION_DIRS = sorted([*INPUT.glob("*/extensions"), *INPUT.glob("*/*/extensions")])


def find_files(pattern):
    """Fichiers correspondant à `pattern` (ex. "dom_ghg_emissions/d_cba.pkl",
    "*raw_materials/M.pkl") dans chacun des dossiers EXTENSION_DIRS."""
    return [p for ext_dir in EXTENSION_DIRS for p in ext_dir.glob(pattern)]


def collapse_to_single_row(df, label):
    """Effondre toutes les lignes de df en une seule, nommée `label` (ex. "CO2", "RMC")."""
    collapsed = df.sum().rename(label).to_frame().T
    collapsed.index.name = "indicator"
    return collapsed


def add_sum_row(df, label):
    """Réduit l'index à son dernier niveau (le "sector", renommé "indicator")
    si besoin, en conservant toutes les lignes, et ajoute une ligne `label`
    égale à leur somme (ex. "RMC" = somme de toutes les matières premières)."""
    if df.index.nlevels > 1:
        df = df.set_index(df.index.get_level_values(-1))
    df.index.name = "indicator"

    if label in df.index:
        return df

    total = df.sum().rename(label).to_frame().T
    total.index.name = "indicator"
    return pd.concat([df, total])


def save_pickle(df, out_path, note=""):
    """Écrit df en pickle sous out_path (en créant les dossiers manquants) et log le résultat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    label = f"Done ({note})" if note else "Done"
    print(f"{label}: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")


def compute_f_x_dom_from_f_y_f_z(extension_name, label, keep_detail=False):
    """Recalcule F_x_dom.pkl (empreinte production, part domestique) à partir de
    F_Y.pkl + F_Z.pkl, pour une extension qui n'a pas de F_x_dom.pkl fourni
    directement en amont. Si keep_detail=False, effondre l'index en une seule
    ligne `label` (ghg_combustion : "CO2"). Si keep_detail=True, conserve une
    ligne par matière première (indexée par "indicator", au niveau "sector")
    et ajoute une ligne `label` = somme (raw_materials : "RMC")."""
    paths_by_scenario = defaultdict(list)
    for in_path in sorted([
        *find_files(f"dom_{extension_name}/F_Y.pkl"),
        *find_files(f"dom_{extension_name}/F_Z.pkl"),
    ]):
        paths_by_scenario[in_path.parent.parent].append(in_path)

    for scenario_dir, paths in paths_by_scenario.items():
        f_x_dom = pd.DataFrame()
        for in_path in paths:
            with open(in_path, "rb") as f:
                df = pickle.load(f)
            df = df[df.index.get_level_values("origin") == "domestic"]
            df = df.droplevel("origin")

            if keep_detail:
                if df.index.nlevels > 1:
                    df = df.set_index(df.index.get_level_values(-1))
                    df.index.name = "indicator"
                df = df.sum(axis=1).groupby(level="indicator").sum().to_frame()
            elif label not in df.index:
                df = collapse_to_single_row(df, label).sum(axis=1).to_frame()

            f_x_dom = f_x_dom.add(df, fill_value=0)

        if keep_detail and label not in f_x_dom.index:
            f_x_dom = add_sum_row(f_x_dom, label)

        out_path = OUTPUT / scenario_dir.relative_to(INPUT) / f"dom_{extension_name}" / "F_x_dom.pkl"
        save_pickle(f_x_dom, out_path)


def reformat_s_like_f_x_dom(extension_name, label, keep_detail=False):
    """Reformate les lignes de S_Y.pkl / S_Z.pkl comme celles de F_x_dom.pkl :
    part domestique uniquement, puis soit effondrée en une seule ligne `label`
    (ghg_combustion : "CO2"), soit une ligne par matière première + une ligne
    `label` = somme (raw_materials : "RMC"). Les colonnes sont conservées."""
    for in_path in sorted([
        *find_files(f"dom_{extension_name}/S_Y.pkl"),
        *find_files(f"dom_{extension_name}/S_Z.pkl"),
    ]):
        with open(in_path, "rb") as f:
            df = pickle.load(f)

        if "origin" in df.index.names:
            df = df[df.index.get_level_values("origin") == "domestic"]
            df = df.droplevel("origin")

        if keep_detail:
            df = add_sum_row(df, label)
        elif label not in df.index:
            df = collapse_to_single_row(df, label)

        save_pickle(df, OUTPUT / in_path.relative_to(INPUT))


# --- 1a. dom_ghg_emissions : drop ghg_combustion, droplevel source, rename gas -> indicator ---
for in_path in sorted([
    *find_files("dom_ghg_emissions/d_cba_k.pkl"),
    *find_files("dom_ghg_emissions/d_cba.pkl"),
    *find_files("dom_ghg_emissions/F_x_dom.pkl"),
    *find_files("dom_ghg_emissions/M.pkl"),
    *find_files("dom_ghg_emissions/M_k.pkl"),
    *find_files("dom_ghg_emissions/S_x_dom.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "source" in df.index.names:
        df = df.drop(index="ghg_combustion", level="source")
        df = df.droplevel("source")
        df.index.names = ["indicator" if n == "gas" else n for n in df.index.names]

    save_pickle(df, OUTPUT / in_path.relative_to(INPUT))

# --- 1b. imp_ghg_emissions : sum sources by gas, rename gas -> indicator ---
for in_path in sorted([
    *find_files("imp_ghg_emissions/d_cba_k.pkl"),
    *find_files("imp_ghg_emissions/d_cba.pkl"),
    *find_files("imp_ghg_emissions/M.pkl"),
    *find_files("imp_ghg_emissions/M_k.pkl"),
    *find_files("imp_ghg_emissions/M_RoW.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "source" in df.index.names:
        df = df.groupby(level="gas").sum()
        df.index.name = "indicator"

    save_pickle(df, OUTPUT / in_path.relative_to(INPUT))

# --- 2. dom_ghg_combustion : sum -> "CO2", create imp_ghg_combustion with zeros ---
for in_path in sorted([
    *find_files("dom_ghg_combustion/d_cba_k.pkl"),
    *find_files("dom_ghg_combustion/d_cba.pkl"),
    *find_files("dom_ghg_combustion/M.pkl"),
    *find_files("dom_ghg_combustion/M_k.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "CO2" not in df.index:
        df = collapse_to_single_row(df, "CO2")

    out_path = OUTPUT / in_path.relative_to(INPUT)
    save_pickle(df, out_path)

    imp_path = Path(str(out_path).replace("dom_ghg_combustion", "imp_ghg_combustion"))
    save_pickle(df * 0, imp_path, note="imp zeros")

compute_f_x_dom_from_f_y_f_z("ghg_combustion", "CO2")
reformat_s_like_f_x_dom("ghg_combustion", "CO2")

# --- 3. *_raw_materials : conserve toutes les lignes, ajoute "RMC" = somme ---
for in_path in sorted([
    *find_files("*raw_materials/d_cba_k.pkl"),
    *find_files("*raw_materials/d_cba.pkl"),
    *find_files("*raw_materials/M.pkl"),
    *find_files("*raw_materials/M_k.pkl"),
    *find_files("imp_raw_materials/M_RoW.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    df = add_sum_row(df, "RMC")

    save_pickle(df, OUTPUT / in_path.relative_to(INPUT))

compute_f_x_dom_from_f_y_f_z("raw_materials", "RMC", keep_detail=True)
reformat_s_like_f_x_dom("raw_materials", "RMC", keep_detail=True)
