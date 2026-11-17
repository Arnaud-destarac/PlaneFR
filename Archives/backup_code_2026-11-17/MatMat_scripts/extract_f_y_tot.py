"""
Extract F_Y_tot (final demand fluxes summed over the Y-category level,
keeping only the region level in columns) for each EXIOBASE3 EEIO
extension, and save it independently of the adapter's normal output.

F_Y is not part of the accounts data model produced by the exiobase3_eeio
adapter (Exiobase3EEIO._fill_extensions only copies F and unit into the
local Accounts), so it never reaches the adapter's usual output. This
script re-runs the same load + harmonization steps as the adapter
(load() and _concat_extension_in_pymrio_format()), then reuses each
extension's F_Y the same way as its F:
- pass-through extensions (raw EXIOBASE satellite accounts) directly take
  the F_Y of their raw account;
- custom extensions (ghg_emissions, raw_materials) reuse their extractor by
  substituting F_Y in place of F: these extractors filter/aggregate purely
  on the row index (stressor label, identical for F and F_Y) or on the
  untouched unit table, so this yields F_Y processed the exact same way as
  the usual F output.

All extensions are computed with a single EXIOBASE version, WORLD_VERSION,
shared with the rest of the pipeline: it is read from the
PLANEFR_EXIOBASE_VERSION environment variable (set by run_full_pipeline.ps1
/ run_pipeline_from_engine.ps1), 3.11.2 by default. The extension list is
the one of the exiobase3_eeio settings (EXIOBASE_SETTINGS_FILE), so that
F_Y_tot is produced for every extension extracted upstream.

Usage: run from the project root so config.toml (data_dir) is picked up:
    uv run python scripts/extract_f_y_tot.py
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pymrio

MATMAT_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(MATMAT_SRC))

from matmat.workflows.adapters.accounts.exiobase3_eeio.core import (
    Exiobase3EEIO,
)
from matmat.workflows.adapters.accounts.exiobase3_eeio.identity import (
    Exiobase3EEIOIdentity,
)
import matmat.utils.constants as cst
from matmat.utils import config

BASE_YEARS = [2019, 2015]
SYSTEM = "pxp"
# Same extensions as the exiobase3_eeio step of the pipeline (may be ["all"])
EXIOBASE_SETTINGS_FILE = (
    Path(__file__).resolve().parent.parent
    / "data" / "4-PlaneFR" / "Settings" / "1-exiobase3_eeio"
    / "exiobase3_eeio_settings.json"
)
with open(EXIOBASE_SETTINGS_FILE, encoding="utf-8") as f:
    EXTENSION_NAMES = json.load(f)["extension_names"]

# Relative to data_dir (see config.toml).
PATH_IN = r".\01-sources_public\exiobase"

# Final destination, one folder per base_year, matching the
# {extensions}/{extension_name}/F_Y_tot.pkl layout.
WORLD_PATH_OUT = r".\4-PlaneFR\Outputs\World"
WORLD_VERSION = os.environ.get("PLANEFR_EXIOBASE_VERSION", "3.11.2")

FILE_NAME = "F_Y_tot"


def extract_f_y_tot(
    extension_name: str, adapter: Exiobase3EEIO
) -> pd.DataFrame | None:
    """
    Compute F_Y_tot for one extension.

    For a custom extension, F_Y is substituted in place of F before calling
    its extractor, so its filtering/conversion logic (driven by the row
    index or by the unchanged unit table) processes F_Y exactly like it
    would process F. For a pass-through extension, the F_Y of the raw
    EXIOBASE satellite account is used as-is. The result is then summed
    over the Y-category level, keeping only the region level in columns.

    Returns:
        pd.DataFrame | None: F_Y_tot, or None if F_Y is entirely empty for
            this extension.
    """
    if extension_name in Exiobase3EEIO.CUSTOM_EXTRACTORS:
        extensions_in = adapter.get_processed_data(
            adapter.KEY_PYMRIO_ACCOUNTS
        ).extensions
        substituted = pymrio.Extension(
            name=extensions_in.name,
            F=extensions_in.F_Y,
            F_Y=extensions_in.F_Y,
            unit=extensions_in.unit,
        )
        extractor_cls = Exiobase3EEIO.CUSTOM_EXTRACTORS[extension_name]
        f_y = extractor_cls(substituted).extract().F
    else:
        f_y = adapter._raw_extensions[extension_name].F_Y

    if f_y is None or f_y.isna().all().all():
        return None

    return f_y.T.groupby(level=cst.IDX_REGION).sum().T


def compute_f_y_tot(
    version: str, base_year: int, extension_names: list[str] = EXTENSION_NAMES
) -> dict[str, pd.DataFrame]:
    """
    Run the adapter's load + harmonization steps for one (version,
    base_year) and compute F_Y_tot for every extension in extension_names.

    Returns:
        dict[str, pd.DataFrame]: F_Y_tot per extension name (extensions
            whose F_Y is entirely empty are omitted).
    """
    identity = Exiobase3EEIOIdentity(
        path_in=PATH_IN,
        # path_out is never used: save() is never called, and the folder
        # name derivation this script relies on happens in the World
        # output layout instead.
        path_out=WORLD_PATH_OUT,
        clean_path_out=False,
        export_format=[cst.FORMAT_PICKLE],
        version=version,
        system=SYSTEM,
        base_year=base_year,
        extension_names=extension_names,
    )
    adapter = Exiobase3EEIO(id_=identity, no_confirm=True)
    adapter.load()
    adapter._resolve_extension_names()
    adapter._concat_extension_in_pymrio_format()

    f_y_tot_by_extension = {}
    for extension_name in adapter._id.extension_names:
        f_y_tot = extract_f_y_tot(extension_name, adapter)
        if f_y_tot is None:
            print(f"Skip {extension_name}: F_Y is empty")
            continue
        f_y_tot_by_extension[extension_name] = f_y_tot

    return f_y_tot_by_extension


def send_f_y_tot_to_world_outputs(
    base_year: int, f_y_tot_by_extension: dict[str, pd.DataFrame]
) -> None:
    """
    Save F_Y_tot to the final World outputs layout, for each extension.
    """
    for extension_name, f_y_tot in f_y_tot_by_extension.items():
        ext_dir = os.path.join(
            config.DATA_DIR,
            WORLD_PATH_OUT,
            f"Base_year_{base_year}",
            "extensions",
            extension_name,
        )
        os.makedirs(ext_dir, exist_ok=True)
        dest_file = os.path.join(ext_dir, f"{FILE_NAME}.pkl")
        f_y_tot.to_pickle(dest_file)
        print(f"Saved {extension_name}: {f_y_tot.shape} -> {dest_file}")


def main() -> None:
    for base_year in BASE_YEARS:
        print(f"=== version={WORLD_VERSION} / base_year={base_year} ===")
        f_y_tot_by_extension = compute_f_y_tot(WORLD_VERSION, base_year)

        send_f_y_tot_to_world_outputs(base_year, f_y_tot_by_extension)


if __name__ == "__main__":
    main()
