import pickle
import pandas as pd

PATHS = [
    "data/3.10.2/base_year_2019/imp_ghg_emissions/d_cba_k.pkl",
    "data/3.10.2/base_year_2019/dom_ghg_emissions/d_cba_k.pkl",
]

for path in PATHS:
    with open(path, "rb") as f:
        df = pickle.load(f)

    # Drop rows where source == 'ghg_combustion'
    df = df.drop(index="ghg_combustion", level="source")

    # Drop 'source' index level (only 'ghg_other' remains, redundant)
    df = df.droplevel("source")

    # Rename index level 'gas' -> 'indicator'
    df.index.names = ["indicator" if n == "gas" else n for n in df.index.names]

    with open(path, "wb") as f:
        pickle.dump(df, f)

    print(f"Done: {path}  shape={df.shape}  index={df.index.names}")
