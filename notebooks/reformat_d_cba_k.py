import pickle
from collections import defaultdict
from pathlib import Path
import pandas as pd

INPUT = Path(r"C:\Users\Arnaud\Documents\CIRED\MatMat\260609-datapack\data\4-PlanéFR\Outputs\France")
OUTPUT = Path(r"C:\Users\Arnaud\Documents\PlaneFR\data\3.10.2")

# --- 1a. dom_ghg_emissions : drop ghg_combustion, droplevel source, rename gas -> indicator ---
for in_path in sorted([
    *INPUT.rglob("dom_ghg_emissions/d_cba_k.pkl"),
    *INPUT.rglob("dom_ghg_emissions/d_cba.pkl"),
    *INPUT.rglob("dom_ghg_emissions/F_x_dom.pkl")
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "source" in df.index.names:
        df = df.drop(index="ghg_combustion", level="source")
        df = df.droplevel("source")
        df.index.names = ["indicator" if n == "gas" else n for n in df.index.names]

    out_path = OUTPUT / in_path.relative_to(INPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")

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

    out_path = OUTPUT / in_path.relative_to(INPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")

# --- 2. dom_ghg_combustion : sum -> "CO2", create imp_ghg_combustion with zeros ---
for in_path in sorted([
    *INPUT.rglob("dom_ghg_combustion/d_cba_k.pkl"),
    *INPUT.rglob("dom_ghg_combustion/d_cba.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "CO2" not in df.index:
        df = df.sum().rename("CO2").to_frame().T
        df.index.name = "indicator"
    
    out_path = OUTPUT / in_path.relative_to(INPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    imp_path = Path(str(out_path).replace("dom_ghg_combustion", "imp_ghg_combustion"))
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(imp_path, "wb") as f:
        pickle.dump(df * 0, f)

    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")
    print(f"Done (imp zeros): {imp_path.relative_to(OUTPUT)}")

# calcul de F_x_dom_tot à partir de F_Y et F_Z
paths_by_scenario = defaultdict(list)
for in_path in sorted([
    *INPUT.rglob("dom_ghg_combustion/F_Y.pkl"),
    *INPUT.rglob("dom_ghg_combustion/F_Z.pkl"),
]):
    paths_by_scenario[in_path.parent.parent].append(in_path)

for scenario_dir, paths in paths_by_scenario.items():
    F_x_dom = pd.DataFrame()
    for in_path in paths:
        with open(in_path, "rb") as f:
            df = pickle.load(f)
        df = df[df.index.get_level_values("origin") == "domestic"]
        df = df.droplevel("origin")
        if "CO2" not in df.index:
            df = df.sum().rename("CO2").to_frame().T
            df.index.name = "indicator"
            df = df.sum(axis=1).to_frame()
        F_x_dom = F_x_dom.add(df, fill_value=0)

    out_path = OUTPUT / scenario_dir.relative_to(INPUT) / "dom_ghg_combustion" / "F_x_dom.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(F_x_dom, f)
    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={F_x_dom.shape}  index={F_x_dom.index.names}")

# --- 3. *_raw_materials : sum -> "RMC" ---
for in_path in sorted([
    *INPUT.rglob("*raw_materials/d_cba_k.pkl"),
    *INPUT.rglob("*raw_materials/d_cba.pkl"),
]):
    with open(in_path, "rb") as f:
        df = pickle.load(f)

    if "RMC" not in df.index:
        df = df.sum().rename("RMC").to_frame().T
        df.index.name = "indicator"

    out_path = OUTPUT / in_path.relative_to(INPUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(df, f)

    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={df.shape}  index={df.index.names}")

# calcul de F_x_dom_tot à partir de F_Y et F_Z
paths_by_scenario = defaultdict(list)
for in_path in sorted([
    *INPUT.rglob("dom_raw_materials/F_Y.pkl"),
    *INPUT.rglob("dom_raw_materials/F_Z.pkl"),
]):
    paths_by_scenario[in_path.parent.parent].append(in_path)

for scenario_dir, paths in paths_by_scenario.items():
    F_x_dom = pd.DataFrame()
    for in_path in paths:
        with open(in_path, "rb") as f:
            df = pickle.load(f)
        df = df[df.index.get_level_values("origin") == "domestic"]
        df = df.droplevel("origin")
        if "RMC" not in df.index:
            df = df.sum().rename("RMC").to_frame().T
            df.index.name = "indicator"
            df = df.sum(axis=1).to_frame()
        F_x_dom = F_x_dom.add(df, fill_value=0)

    out_path = OUTPUT / scenario_dir.relative_to(INPUT) / "dom_raw_materials" / "F_x_dom.pkl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(F_x_dom, f)
    print(f"Done: {out_path.relative_to(OUTPUT)}  shape={F_x_dom.shape}  index={F_x_dom.index.names}")