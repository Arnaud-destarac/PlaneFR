"""
Reformatage ponctuel des extensions France (sortie brute du pipeline CIRED/MatMat,
dossier INPUT) vers data/3.10.2/ :
- ghg_emissions : retire la part ghg_combustion (traitée à part ci-dessous),
  renomme le niveau d'index "gas" en "indicator".
- ghg_combustion : effondre en une seule ligne "CO2" ; imp_ghg_combustion n'existe
  pas en amont, on le crée à zéro (les émissions de combustion sont par
  construction 100% domestiques).
- raw_materials : effondre en une seule ligne "RMC".
- F_x_dom de ghg_combustion et raw_materials : ces deux extensions n'ont pas de
  F_x_dom.pkl fourni en amont ; on le recalcule à partir de F_Y.pkl + F_Z.pkl
  (part domestique uniquement), effondré de la même façon.

Script à usage unique lié à un chemin source externe à ce repo (poste du
CIRED) : nettoyé pour la lisibilité (les idiomes "effondrer en une ligne" et
"recalculer F_x_dom" étaient dupliqués), pas pour être généralisé ou re-testé
automatiquement.
"""

import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd

INPUT = Path(r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\France")
OUTPUT = Path(r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2")


def collapse_to_single_row(df, label):
    """Effondre toutes les lignes de df en une seule, nommée `label` (ex. "CO2", "RMC")."""
    collapsed = df.sum().rename(label).to_frame().T
    collapsed.index.name = "indicator"
    return collapsed


def save_pickle(df, out_path, note=""):
    """Écrit df en pickle sous out_path (en créant les dossiers manquants) et log le résultat."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)
    label = f"Done ({note})" if note else "Done"
    print(f"{label}: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")


def compute_f_x_dom_from_f_y_f_z(extension_name, label):
    """Recalcule F_x_dom.pkl (empreinte production, part domestique) à partir de
    F_Y.pkl + F_Z.pkl, pour une extension dont l'index doit être effondré en une
    seule ligne `label` — utilisé pour ghg_combustion ("CO2") et raw_materials
    ("RMC"), qui n'ont pas de F_x_dom.pkl fourni directement en amont."""
    paths_by_scenario = defaultdict(list)
    for in_path in sorted([
        *INPUT.rglob(f"dom_{extension_name}/F_Y.pkl"),
        *INPUT.rglob(f"dom_{extension_name}/F_Z.pkl"),
    ]):
        paths_by_scenario[in_path.parent.parent].append(in_path)

    for scenario_dir, paths in paths_by_scenario.items():
        f_x_dom = pd.DataFrame()
        for in_path in paths:
            with open(in_path, "rb") as f:
                df = pickle.load(f)
            df = df[df.index.get_level_values("origin") == "domestic"]
            df = df.droplevel("origin")
            if label not in df.index:
                df = collapse_to_single_row(df, label).sum(axis=1).to_frame()
            f_x_dom = f_x_dom.add(df, fill_value=0)

        out_path = OUTPUT / scenario_dir.relative_to(INPUT) / f"dom_{extension_name}" / "F_x_dom.pkl"
        save_pickle(f_x_dom, out_path)


# --- 1a. dom_ghg_emissions : drop ghg_combustion, droplevel source, rename gas -> indicator ---
for in_path in sorted([
    *INPUT.rglob("dom_ghg_emissions/d_cba_k.pkl"),
    *INPUT.rglob("dom_ghg_emissions/d_cba.pkl"),
    *INPUT.rglob("dom_ghg_emissions/F_x_dom.pkl"),
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
    *INPUT.rglob("imp_ghg_emissions/d_cba_k.pkl"),
    *INPUT.rglob("imp_ghg_emissions/d_cba.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "source" in df.index.names:
        df = df.groupby(level="gas").sum()
        df.index.name = "indicator"

    save_pickle(df, OUTPUT / in_path.relative_to(INPUT))

# --- 2. dom_ghg_combustion : sum -> "CO2", create imp_ghg_combustion with zeros ---
for in_path in sorted([
    *INPUT.rglob("dom_ghg_combustion/d_cba_k.pkl"),
    *INPUT.rglob("dom_ghg_combustion/d_cba.pkl"),
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

# --- 3. *_raw_materials : sum -> "RMC" ---
for in_path in sorted([
    *INPUT.rglob("*raw_materials/d_cba_k.pkl"),
    *INPUT.rglob("*raw_materials/d_cba.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "RMC" not in df.index:
        df = collapse_to_single_row(df, "RMC")

    save_pickle(df, OUTPUT / in_path.relative_to(INPUT))

compute_f_x_dom_from_f_y_f_z("raw_materials", "RMC")
