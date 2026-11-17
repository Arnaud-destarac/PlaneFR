# PlaneFR module — technical reference documentation

This document describes the PlaneFR module, which assesses the environmental pressures of France against the planetary boundaries using the MatMat input-output model. 

It describes the state of the code on 17 November 2026.

## Contents

1. [What the module does](#1-what-the-module-does)
2. [Code and data layout](#2-code-and-data-layout)
3. [Installation and prerequisites](#3-installation-and-prerequisites)
4. [Changes to the MatMat source code](#4-changes-to-the-matmat-source-code)
5. [Computation chain in matmat-ademe](#5-computation-chain-in-matmat-ademe)
6. [Building the scenario shocks](#6-building-the-scenario-shocks)
7. [Post-processing and figures: Module_PlaneFR](#7-post-processing-and-figures-module_planefr)
8. [Breakdown by household category: ademe_eemrio](#8-breakdown-by-household-category-ademe_eemrio)
9. [Data and parameter files](#9-data-and-parameter-files)
10. [Summary of assumptions](#10-summary-of-assumptions)
11. [User guide](#11-user-guide)
12. [Glossary](#12-glossary)

---

## 1. What the module does

For France, the module computes the consumption footprint and the territorial pressures on nine planetary-boundary indicators, then compares them with a share of the global budget allocated to France. It does so for the base year and for scenarios to 2050.

The computation relies on three building blocks, spread over three repositories:

| Block | Repository | Role |
|---|---|---|
| Model | `matmat-ademe` (fork of MatMat) | Extracts the EXIOBASE accounts, builds the France accounts, applies the scenario shocks and computes the footprints. |
| Post-processing | `Module_PlaneFR` | Converts the EXIOBASE extensions into planetary-boundary indicators, compares them with the thresholds and produces the figures. |
| Households | `ademe_eemrio` | Splits the footprint between household categories, using the 2017 Budget de famille survey. |

### Indicators

The code distinguishes two levels, which helps when reading the scripts:

- the **extension** is the MatMat satellite account (`land`, `water`, `nutrients`…). In `planefr_lib` it is called `lp_name` and corresponds to the "EXIOBASE groups" column of `characterization_factors.xlsx`;
- the **Indicator** is what is shown in the figures ("N surplus", "GHG emissions"…), i.e. the "Indicators" column of `characterization_factors.xlsx` and a column of `thresholds.xlsx`. An indicator can aggregate several extensions.

| Indicator | Extensions used | Figure unit |
|---|---|---|
| GHG emissions | `ghg_emissions`, `ghg_combustion` | MtCO2eq |
| N surplus | `nutrients`, `air_emissions` | Gg |
| P surplus | `nutrients` | Gg |
| Blue water consumption | `water` | Md m³ |
| Cropland use | `land` | Md m² |
| Agricultural land | `land` | Md m² |
| Raw material consumption | `raw_materials` | Mt |
| Biodiversity loss - PDF | `air_emissions`, `ghg_emissions`, `ghg_combustion`, `land`, `nutrients`, `water` | Md PDF.m2.yr |
| Biodiversity loss - HANPP | `land`, `raw_materials` | MtC |

The `energy` and `deforestation` extensions are extracted and computed by the model, but no indicator uses them at present.

### Scenarios

| Name | Content | Scope of the shock |
|---|---|---|
| `Base_year_2015`, `Base_year_2019` | Base years, no shock | — |
| `S1_2050` to `S4_2050`, `TEND_2050` | ADEME Transition(s) 2050 scenarios and trend scenario | French economy |
| `SSP2_2050`, `SSP2_19_2050`, `SSP2_19_Sus_2050` | Scenarios of the IMAGE model | Rest of the world |
| `S1+` | S1 applied to accounts whose rest of the world follows `SSP2_19_Sus` | France and rest of the world |
| suffix `_agri` | Same scenarios, pressures restricted to the agri-food system | — |
| `2019_W`, `2019_EU27`, `2019_<code>` | World, EU27 and each of the 49 EXIOBASE regions in 2019 | — |

A 2050 France scenario therefore combines two independent choices: the French scenario (S1 to S4, TEND) and the state of the rest of the world (unchanged from the base year, or projected along an IMAGE scenario).

### Two accounting perspectives

- **CBA** (consumption-based accounting): pressures exerted anywhere in the world to meet French final demand. This is the footprint, the sum of a domestic part and an imported part.
- **PBA** (production-based accounting): pressures exerted on French territory, by production and directly by final demand.

---

## 2. Code and data layout

The links in this document and several paths in the code assume the following local tree:

```
Code/
├── MatMat/
│   ├── matmat-ademe/        fork of MatMat: modified source code, scripts, data, settings
│   └── datapack/            MatMat training environment (matmat 0.9.0b0 wheel)
├── Module_PlaneFR/          post-processing, parameter data, figures
└── Github/
    └── ademe_eemrio/        breakdown by household category
```

| Repository | Remote | Branch | Revision described |
|---|---|---|---|
| `matmat-ademe` | `Arnaud-destarac/matmat-ademe`, fork of `AntoineTeixeira/matmat-ademe` | `main` | `04eab49`, derived from upstream commit `bce703f` (MatMat 0.9.0-beta)
| `Module_PlaneFR` | `Arnaud-destarac/PlaneFR` | `main` | `ba86f8c8`
| `ademe_eemrio` | `CIRED/ademe_eemrio` | `ademe_eemrio_dev` | `5319dd4`


### Overall flow

```mermaid
flowchart TD
    EXIO["EXIOBASE 3.11.2<br/>pxp archives 2015 and 2019"] --> AD["exiobase3_eeio adapter<br/>(modified)"]
    AD --> WB["World accounts<br/>base year"]
    IMAGE["IMAGE scenarios<br/>+ matching workbook"] --> SSP["Rest-of-world shocks<br/>dS and dA"]
    WB --> ENGW["eeio engine, World"]
    SSP --> ENGW
    ENGW --> WS["World accounts<br/>base and 2050"]
    WS --> SNAC["gmrio_to_snac_s pipeline"]
    SNAC --> CAL["update_calib_fr pipeline<br/>(datapack environment)"]
    CALIB["ADEME France calibration<br/>3.10.2_France_2015 / 2019"] --> CAL
    CAL --> FR["France accounts"]
    T2050["Transition(s) 2050 shocks<br/>dA, dY, dY_k"] --> ENGF["eeio engine, France"]
    MAN["Manual shocks<br/>my_manual_param.xlsx, dS"] --> ENGF
    FR --> ENGF
    ENGF --> OUT["Results by scenario<br/>Module_PlaneFR/data/"]
    WS --> REF["Reformatting scripts"]
    REF --> OUT
    OUT --> LIB["planefr_lib<br/>footprints, thresholds, figures"]
    PAR["Characterization factors,<br/>thresholds, budget shares"] --> LIB
    OUT --> BDF["ademe_eemrio<br/>household breakdown"]
    BDF --> LIB
```

### Where the files are

In `matmat-ademe`, everything specific to PlanéFR is in the `planeFR/` folder, which is versioned: the pipeline scripts and their parameters. The data, accounts and outputs stay in `data/4-PlaneFR/`, which is excluded from git like the whole `data/` folder (`**/data/` in `.gitignore`).

```
matmat-ademe/
├── planeFR/                  versioned
│   ├── scripts/              pipeline scripts (Python and PowerShell), one folder per step
│   │   ├── _common/              shared paths and helpers
│   │   ├── 0-chains/             chains of steps (run_full_pipeline.ps1...)
│   │   ├── 1-exiobase3_eeio/     EXIOBASE adapter, World engine, F_Y extraction
│   │   ├── 2-gmrio_to_snac_s/    France SNAC accounts, shock extension folders
│   │   ├── 3-update_calib_fr/    merge with the France calibration
│   │   ├── 4-engine/             France and World engines
│   │   ├── 5-reformat/           World/Europe/region aggregates, harmonized extensions
│   │   ├── 6-scenario_groups/    display groups
│   │   ├── a-manual_shocks/      manual shocks on French domestic pressures
│   │   ├── b-ssp_shocks/         rest-of-world shocks from IMAGE
│   │   └── c-agri_shocks/        agri-food variants of the shocks
│   ├── parameters/           settings templates, bridge matrices, manual shocks
│   │   ├── 1-exiobase3_eeio/ … 4-engine/   templates and bridges of each MatMat step
│   │   └── a-manual_shocks/      my_manual_param.xlsx
│   └── .gitignore
└── data/4-PlaneFR/           not versioned
    ├── 1-accounts/
    ├── 2-shocks/
    ├── Outputs/
    ├── Outputs_inter/
    └── Settings_run/         settings files generated at each run
```

| Location | Content | Versioned |
|---|---|---|
| `matmat-ademe/src/matmat/` | MatMat source code, including the five modified files | yes |
| `matmat-ademe/planeFR/scripts/` | 13 Python scripts and 12 PowerShell scripts of the PlaneFR pipeline, in step folders | yes |
| `matmat-ademe/planeFR/parameters/` | Settings templates (JSON), bridge matrices, manual shocks | yes |
| `matmat-ademe/data/01-sources_public/exiobase/<version>/` | EXIOBASE archives `IOT_<year>_pxp.zip` | no |
| `matmat-ademe/data/4-PlaneFR/Settings_run/` | Settings files generated from the templates at each run | no |
| `matmat-ademe/data/4-PlaneFR/1-accounts/` | EXIOBASE accounts in MatMat format | no |
| `matmat-ademe/data/4-PlaneFR/2-shocks/` | Scenario shocks | no |
| `matmat-ademe/data/4-PlaneFR/Outputs/`, `Outputs_inter/` | Computed accounts (World, France) and intermediate outputs | no |
| `Module_PlaneFR/data/` | Post-processing parameters and model results by scenario | yes |
| `Module_PlaneFR/notebooks/planefr_lib/` | Post-processing library | yes |
| `Module_PlaneFR/notebooks/*.ipynb` | Figure notebooks | yes |
| `Module_PlaneFR/figures/` | Figures and workbooks produced | yes |

The parameters folder must not be named `settings` or placed under `data/`: both are excluded by MatMat's `.gitignore` (the `settings/` rule applies at any depth and, on Windows, regardless of case).

> **Possible improvement — results in git.** `Module_PlaneFR` versions its model results (`data/3.11.2/`, about 8,800 pickles), which can all be regenerated. Git handles large binary files poorly: the repository grows at each run. Git LFS, or publishing the results separately (e.g. on Zenodo, as MatMat does), would keep it light.

---

## 3. Installation and prerequisites

The project uses four Python environments, because the repositories do not have the same dependencies.

| Environment | Python | Key dependencies | Used for |
|---|---|---|---|
| `matmat-ademe/.venv` (managed by `uv`) | 3.11 | pandas 2.2.2, pymrio 0.5.4, scipy 1.17.1 | Adapter, `gmrio_to_snac_s` pipeline, engine, Python pipeline scripts |
| `datapack/.venv` (managed by `uv`) | 3.11 | wheel `matmat-0.9.0b0`, pandas 2.2.2 | `update_calib_fr` pipeline only |
| Module_PlaneFR (`requirements.txt`) | 3.14 | pandas 3.0.1, matplotlib 3.10.8, pymrio 0.6.3, country_converter 1.3.2 | Notebooks, `planefr_lib`, `data/budget_shares.py` |
| ademe_eemrio (`py -3` launcher) | 3.x | pandas, openpyxl, jupyter, nbconvert | Budget de famille chain |

The `update_calib_fr` pipeline does not exist in the public `matmat-ademe` repository: its `src/matmat/workflows/pipelines/accounts/` folder only contains `aggregation` and `gmrio_to_snac_s`. It is only available in the `matmat-0.9.0b0` wheel shipped with the `datapack`. The scripts therefore run this step from the `datapack` folder, while reading the settings of `matmat-ademe`.

The pandas version gap between environments has a practical consequence: a pickle written with pandas 3 cannot be read with pandas 2 when it contains strings of type `StringDtype`. § 4.4 describes the fix.

> **Possible improvement — a single environment.** Porting `update_calib_fr` into `matmat-ademe` would remove the detour through the `datapack` environment, and aligning the pandas versions of `matmat-ademe` and `Module_PlaneFR` would make the pickle workaround of § 4.4 unnecessary. The pipeline would then run in one environment.

### Configuration

Both `config.toml` files (`matmat-ademe` and `datapack`) must point to the same data folder, that of `matmat-ademe`:

```toml
[run]
data_dir = "C:/Users/<user>/.../Code/MatMat/matmat-ademe/data"
settings_dir = "C:/Users/<user>/.../Code/MatMat/matmat-ademe/settings"
```

All relative paths in the JSON settings (`.\4-PlaneFR\...`) are resolved against `data_dir`.

### Paths

The pipeline scripts contain no path specific to a machine, so the repository works for anyone who clones it. All the paths are fixed relative to the repository. The Python scripts take them from [planeFR/scripts/_common/planefr_paths.py](../../MatMat/matmat-ademe/planeFR/scripts/_common/planefr_paths.py) and the PowerShell scripts from [planeFR/scripts/_common/planefr_common.ps1](../../MatMat/matmat-ademe/planeFR/scripts/_common/planefr_common.ps1), which both derive them from their own location:

| Constant | Value |
|---|---|
| `PLANEFR_DIR`, `MATMAT_DIR` | `matmat-ademe/planeFR/`, `matmat-ademe/` |
| `PARAMETERS_DIR` | `matmat-ademe/planeFR/parameters/` |
| `MATMAT_DATA_DIR`, `PLANEFR_DATA_DIR` | `matmat-ademe/data/`, `matmat-ademe/data/4-PlaneFR/` |
| `MODULE_PLANEFR_DIR` | `<root>/Module_PlaneFR/`, or the `PLANEFR_MODULE_DIR` environment variable |
| `DATA_VERSION_DIR` | `Module_PlaneFR/data/<PLANEFR_EXIOBASE_VERSION>/` |

`run_engine_batch.ps1` locates `Module_PlaneFR` the same way. The only files holding machine-specific absolute paths are the two `config.toml` files (`data_dir`, `settings_dir`), which are local by design, and the settings files generated in `data/4-PlaneFR/Settings_run/` at each run.

Each Python script of a step folder adds `_common` to its import path (`sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "_common"))`), then imports `planefr_paths` and, if needed, `planefr_utils` (`make_pickle_portable`). `planefr_paths.get_ssp_base_accounts_dir()` gives the base accounts of the SSP shocks (§ 5.3). Each PowerShell script dot-sources `..\_common\planefr_common.ps1`; the chains call the other scripts through `$PlanefrScriptsDir`, e.g. `Join-Path $PlanefrScriptsDir "4-engine\run_engine_batch.ps1"`.

`planefr_lib`, `data/budget_shares.py` and `ademe_eemrio` only use relative paths too, and assume the same tree of § 2: `planefr_lib.config.BDF_FILE` looks in `../Github/ademe_eemrio/`, and `export_impacts_coicop.py` looks in `../../Module_PlaneFR/notebooks`.

### Environment variables

| Variable | Default | Effect |
|---|---|---|
| `PLANEFR_MODULE_DIR` | `<root>/Module_PlaneFR` | Location of `Module_PlaneFR`, if it is not next to the `MatMat` folder. Read by `_common/planefr_paths.py` and `4-engine/run_engine_batch.ps1`. |
| `PLANEFR_EXIOBASE_VERSION` | `3.11.2` | EXIOBASE version read by the Python pipeline scripts and by `planefr_lib.config`. Set by the PowerShell scripts from their `-Version` parameter. |
| `PLANEFR_SCENARIO_GROUP` | `no_SSP` | Scenario group read by default by `planefr_lib`. |
| `PLANEFR_FIGURES_DIR` | `Module_PlaneFR/figures` | Output folder of the figures. Useful to test a notebook without overwriting the existing figures. |
| `PLANEFR_SCENARIO` | `SSP2_19_Sus_2050/S1_2050` | Scenario processed by `export_impacts_coicop.py`. Set by `BdF.ps1`. |

---

## 4. Changes to the MatMat source code

Compared with upstream commit `bce703f`, five files of the `matmat` package are modified. The changes concern the EXIOBASE adapter and the pickle export. The `eeio` engine, the `gmrio_to_snac_s` pipeline and the computation core are not modified.

| File | Change |
|---|---|
| [exiobase3_eeio/core.py](../../MatMat/matmat-ademe/src/matmat/workflows/adapters/accounts/exiobase3_eeio/core.py) | Generic extraction of extensions, `all` keyword, EXIOBASE 3.11.2 support, deterministic extension order |
| [exiobase3_eeio/extractor.py](../../MatMat/matmat-ademe/src/matmat/workflows/adapters/accounts/exiobase3_eeio/extractor.py) | Six specialized extractors replaced by a "pass-through" extractor |
| [exiobase3_eeio/settings.json](../../MatMat/matmat-ademe/src/matmat/workflows/adapters/accounts/exiobase3_eeio/settings.json) | Example settings updated with the new extension names |
| [utils/constants/domain.py](../../MatMat/matmat-ademe/src/matmat/utils/constants/domain.py) | Six extension-name constants removed |
| [utils/formats/core.py](../../MatMat/matmat-ademe/src/matmat/utils/formats/core.py) | Pickles portable across pandas versions |

The fork also adds the `scripts/` folder (§ 5) and training and development notebooks in `notebooks/`, which are not part of the module.

### 4.1 Generic extraction of EXIOBASE extensions

**Before.** The adapter mapped each MatMat extension to a dedicated extraction class (`EXTRACTOR_MAP`, eight entries). Each class filtered the rows of EXIOBASE's merged satellite account with its own rule: label containing "Water", unit equal to "km2", prefix "N - "… Adding an extension required writing a class, and these filters depended on the labels of a given EXIOBASE version.

**After.** Only two extensions keep a dedicated extractor, because they need real processing (conversion to CO2 equivalent, grouping of materials):

```python
# Exiobase3EEIO, core.py
CUSTOM_EXTRACTORS = {
    cst.GHG_EMISSIONS: cls_extractor.GreenhouseGasEmissionsExtractor,
    cst.RAW_MATERIALS: cls_extractor.RawMaterialsExtractor,
}
KEY_ALL_EXTENSIONS = "all"
```

Any other requested extension is the EXIOBASE satellite account of the same name, taken as is by `PassthroughExtractor`:

```python
# extractor.py
class PassthroughExtractor(ExtensionExtractor):
    def __init__(self, extensions: pymrio.Extension, extension_name: str):
        super().__init__(extensions)
        self._extension_name = extension_name

    def extract(self) -> pymrio.Extension:
        extension_out = copy.deepcopy(self._extensions)
        extension_out.name = self._extension_name
        return extension_out
```

Three behavioural changes follow.

1. **Name validation is deferred.** The available satellite accounts are only known once the archive has been read. `_validate_extensions()`, called in the constructor, is replaced by `_resolve_extension_names()`, called at the start of `process()`. Valid names are the two custom extensions and every account returned by `accounts.get_extensions()`. An invalid name raises a `ValueError` listing the names available in the loaded version.
2. **Raw accounts are set aside before merging.** `_concat_extension_in_pymrio_format()` merges all satellite accounts into one, which the custom extractors need. The requested raw accounts are therefore copied beforehand into `self._raw_extensions`, and their index is renamed `extension_category`.
3. **The `["all"]` keyword** in `extension_names` extracts the two custom extensions and every satellite account of the archive.

As a result, the same pressure can appear in two extensions: the custom extractors filter the merged account, which also contains the rows of the pass-through accounts. Agricultural nitrous oxide, for example, appears both in `ghg_emissions` and in `air_emissions`. This overlap causes no double counting as long as no sub-process takes the same pressure from two extensions. That is the case with the current characterization factors: "Biodiversity loss - PDF", which uses both `ghg_emissions` and `air_emissions`, takes no greenhouse gas from `air_emissions`.

Example settings used by the project:

```json
{
  "path_in": ".\\01-sources_public\\exiobase",
  "path_out": ".\\4-PlaneFR\\1-accounts\\Exiobase\\3.11.2_2019",
  "export_format": ["pickle"],
  "version": "3.11.2",
  "system": "pxp",
  "base_year": 2019,
  "extension_names": ["nutrients", "deforestation", "energy", "ghg_emissions",
                      "land", "raw_materials", "water", "air_emissions"],
  "clean_path_out": true
}
```

### 4.2 Extension names changed

Pass-through extensions now carry the name of the EXIOBASE satellite account. The constants `LABOR`, `VALUE_ADDED`, `WATER`, `LAND_USE`, `BIOGEOCHEMICAL` and `ENERGY` of `domain.py` are removed.

| Former MatMat name | New name |
|---|---|
| `labor` | `employment` |
| `value_added` | `factor_inputs` |
| `land_use` | `land` |
| `biogeochemical` | `nutrients` |
| `water`, `energy` | unchanged |

This change is incompatible with settings, bridge matrices and shocks written with the old names.

The content of the extensions changes too. `water` now contains the whole `water` satellite account of EXIOBASE, not only the rows whose label contains "Water". `land` is no longer defined by the "km2" unit.

### 4.3 EXIOBASE 3.11.2 and deterministic order

Version `3.11.2` is added to `SUPPORTED_VERSIONS_TO_DOI` (DOI `10.5281/zenodo.23039074`).

The satellite accounts are now merged in alphabetical order of their name:

```python
extensions = pymrio.concate_extension(
    [getattr(accounts, ext) for ext in sorted(accounts.get_extensions())],
    name=cst.KEY_EXTENSIONS,
)
```

pymrio loads the satellite accounts from a `set`. Without this sort, the row order of the merged account varied from one run to the next.

### 4.4 Pickles portable across pandas versions

`PickleFormat.export()` now casts any `StringDtype` data, in values, index and columns, to `object` before writing:

```python
def export(self, df: pd.DataFrame, path: str, file_name: str):
    df = self._make_pickle_portable(df)
    df.to_pickle(f"{os.path.join(path, file_name)}.{self.EXTENSION}")
```

The serialized form of `StringDtype` is not stable across pandas versions. Since pandas 3.0, strings get this type by default, and a pickle written that way cannot be read with pandas 2.x. Yet the shocks are produced by scripts that may run under pandas 3, then read by the engine under pandas 2.2.2.

The same function is duplicated as `make_pickle_portable` in `planeFR/scripts/_common/planefr_utils.py`, for the scripts that write pickles without going through `PickleFormat`. The scripts could import the MatMat version instead.

### 4.5 Integration work left

Several pipeline scripts repair or complete MatMat outputs after the fact. Moving these treatments into MatMat would make PlanéFR a module of the model rather than a layer on top of it.

| Script | What it works around | Where it would belong |
|---|---|---|
| `extract_f_y_tot.py` | `F_Y` is not part of the accounts written by the adapter; the script replays the adapter by calling its private methods (`_resolve_extension_names`, `_raw_extensions`) | Export of `F_Y` by the `exiobase3_eeio` adapter |
| `reformat_d_cba_monde_europe.py` | Single-level MultiIndex left by `cast_index_to_multiindex()`; World, EU27 and regional aggregates | Index fix in the extraction; aggregation pipeline of MatMat |
| `reformat_d_cba_k.py` | Different formats of the GHG and raw-material extensions of the France calibration | Harmonization of the calibration extensions in `update_calib_fr` |
| `run_update_calib_fr_batch*.ps1` | `update_calib_fr` only exists in the `datapack` wheel | Port of the pipeline to `matmat-ademe` (§ 3) |

---

## 5. Computation chain in matmat-ademe

The pipeline is driven by PowerShell scripts that prepare a settings file for each run, then call the MatMat command line or a Python script. All are in [matmat-ademe/planeFR/scripts/](../../MatMat/matmat-ademe/planeFR/scripts/).

**Settings templates.** The JSON files of `planeFR/parameters/` are templates: the scripts never modify them. For each run, a script reads a template (`Read-SettingsTemplate`), sets the paths of that run (accounts, shock, output), then writes the result under `data/4-PlaneFR/Settings_run/`, with the same relative path (`Write-RunSettings`). That generated file is passed to `matmat.cli`. Both functions are in `planefr_common.ps1`.

In the templates, the token `{parameters}` stands for the `planeFR/parameters/` folder, e.g. `"bridges": "{parameters}\\2-gmrio_to_snac_s\\bridge_matrices.xlsx"`. It is replaced by the absolute path of that folder in the generated file. The other paths of the templates (accounts, shocks, outputs) remain relative to the `data_dir` of `config.toml`, as MatMat expects: both MatMat versions resolve them with `Path(data_dir).joinpath(value)`.

To change a calculation parameter, such as `m_row_mode` or the list of extensions, edit the template: the next runs use it. The paths of accounts, shocks and outputs in the templates are only examples or starting points, overwritten by the scripts.

> **Possible improvement — orchestration.** The PowerShell scripts only run on Windows, whereas MatMat is also developed on Linux, and they duplicate helpers (`Set-YearInPath`, `Add-Suffix`). A single Python command with sub-commands (`full`, `from-engine`, `ssp`, `agri`) would be cross-platform and testable.

### 5.1 Three chains

| Script | Purpose | Prerequisites |
|---|---|---|
| `run_full_pipeline.ps1` | Base years and French scenarios, rest of the world unchanged | EXIOBASE archives, France calibration |
| `run_pipeline_from_engine.ps1` | Same chain from the engine step onwards | France accounts already built |
| `run_all_SSPs.ps1` | French scenarios on a rest of the world projected by IMAGE | `run_full_pipeline.ps1` already run |
| `run_all_SSPs_agri.ps1` | Variant restricted to the agri-food system | `run_all_SSPs.ps1` already run |

Each chain stops at the first failing step. The batch scripts process all their scenarios even when one fails, so that all the failures are reported at once, then return exit code 1 if at least one run failed: the calling chain then stops.

### 5.2 Base chain: `run_full_pipeline.ps1`

| Step | Folder | Script | Processing | Output (under `data/4-PlaneFR/` unless stated) |
|---|---|---|---|---|
| 0 | `1-exiobase3_eeio` | `run_exiobase3_eeio_batch.ps1` | `exiobase3_eeio` adapter for 2015 and 2019, then `eeio` engine on the world accounts | `1-accounts/Exiobase/<v>_<year>/` then `Outputs/World/Base_year_<year>/` |
| 1 | `2-gmrio_to_snac_s` | `run_gmrio_to_snac_s_batch.ps1` | `gmrio_to_snac_s` pipeline: France accounts in SNAC format, imports separated | `Outputs_inter/gmrio_to_snac_s/<v>_pxp_<year>/` |
| 1b | `2-gmrio_to_snac_s` | `rebuild_shock_extensions.py` | Rebuilds the extension folders of the `1-transitions_2050` shocks | `2-shocks/1-transitions_2050/<scenario>/extensions/` |
| 2 | `3-update_calib_fr` | `run_update_calib_fr_batch.ps1` | `update_calib_fr` pipeline: adds the new extensions to the France calibration | `Outputs/France/Base_year_<year>/` |
| 3 | `4-engine` | `run_engine_batch.ps1` | France `eeio` engine: two base years, then one run per accounts × scenario pair | `Module_PlaneFR/data/<v>/...` |
| 4 | `1-exiobase3_eeio` | `extract_f_y_tot.py` | Direct pressures of final demand, by region | `Outputs/World/Base_year_<year>/extensions/<ext>/F_Y_tot.pkl` |
| 5 | `5-reformat` | `reformat_d_cba_monde_europe.py` | World, EU27 and regional aggregates; France `F_Y_tot` | `Module_PlaneFR/data/<v>/2019_*/` and every France scenario |
| 6 | `5-reformat` | `reformat_d_cba_k.py` | Harmonizes the format of the GHG and materials extensions | Files rewritten in place |
| 7 | `a-manual_shocks` | `export_dS.py` | Exports the base-year production and pressures | `Module_PlaneFR/data/S_extraction.xlsx` |
| 8 | `a-manual_shocks` | `compute_ds_x_dom.py` | Computes the manual intensity shocks and applies the `F_Y_tot` shocks | `2-shocks/1-transitions_2050/<scenario>/extensions/dom_<ext>/dS_x_dom.pkl`, `F_Y_tot.pkl` of every scenario |
| 9 | `0-chains` | `run_pipeline_from_engine.ps1` | Second pass: steps 3 to 8 again, the engine applying the up-to-date `dS_x_dom`, then `build_scenario_groups.py` (§ 5.5) | Same as steps 3 to 8, display groups |

**Why a second pass.** Step 8 computes intensity shocks that the engine, at step 3, must apply. But it needs the production of the 2050 scenarios, which is an output of step 3. The first pass therefore applies the `dS_x_dom` of the previous run, which step 1b keeps; the second pass (step 9) applies the ones just computed. One iteration is enough and gives an exact result: production depends on the shocks to the production system (`dA`, `dY`, `dY_k`), not on the intensity shocks.

`-SinglePass` skips step 9. The result is then exact only if neither `my_manual_param.xlsx` nor the accounts or system shocks have changed since the previous run, and only if a previous run exists.

`compute_ds_x_dom.py` deletes a `dS_x_dom` that has become entirely zero (shocks reset to 0 or emptied), and those of extensions that no longer have a sheet in `my_manual_param.xlsx`, so that the engine never applies an outdated shock.

#### Step 0 — EXIOBASE extraction and world accounts

The adapter reads the `IOT_<year>_pxp.zip` archive (product by product, 49 regions, 200 products) and writes MatMat accounts. The `eeio` engine, with the `standard` strategy and no shock (`settings_engine_World.json`), then computes for each extension the intensities `S_x_dom`, the multipliers `M` and the footprint `d_cba` of each region.

#### Step 1 — conversion to SNAC format

The `gmrio_to_snac_s` pipeline, unmodified, turns the world accounts into single-country accounts. The bridge matrix `planeFR/parameters/2-gmrio_to_snac_s/bridge_matrices.xlsx` defines France as the domestic region and aggregates the 48 other regions into one "import" origin.

Each extension is split in two: `dom_<ext>` holds the pressures exerted in France, `imp_<ext>` holds the pressure content of imports (`M_RoW`). This content is computed with the `foreign_global_feedback` mode: the full world production system is used, but the direct pressures of France are set to zero. The trade loops between France and the rest of the world are thus taken into account without counting domestic pressures twice.

The `calculate_accounts` option is `false`: the accounts are computed at the next step.

#### Step 1b — shock extension folders

The `1-transitions_2050` shocks were delivered with a production system at the resolution of the France calibration. For each scenario, `rebuild_shock_extensions.py` creates one `dom_<ext>` folder per extension, with a `detail_levels.xlsx` combining the categories of the extension (from `gmrio_to_snac_s`) and the detail levels of the scenario's system. Without it, the engine raises `MEInconsistentDetailLevels`. The `raw_materials` and `ghg_emissions` extensions, which come from the France calibration, are excluded from this rebuild.

Before rebuilding, the script empties the `extensions` folder of the scenario, except the `dS_*` files (the `dS_x_dom` written by `compute_ds_x_dom.py`), which are kept for the first engine pass. Sub-folders of extensions that are no longer rebuilt are deleted entirely.

#### Step 2 — merging with the France calibration

The `update_calib_fr` pipeline starts from the France accounts calibrated by ADEME, read in `data/3-workshops/1-accounts/MatMat/3.10.2_France_<year>/`, and adds to them the extensions computed at step 1. The production system remains that of the France calibration.

| Origin | Extensions |
|---|---|
| France calibration (`older_accounts`) | `dom_ghg_emissions`, `dom_ghg_combustion`, `dom_raw_materials` |
| EXIOBASE via `gmrio_to_snac_s` (`newer_accounts`) | `dom_` and `imp_` for `water`, `energy`, `land`, `nutrients`, `air_emissions`, `deforestation`; `imp_ghg_emissions`, `imp_raw_materials` |

The new extensions are brought to the resolution of the France calibration by `planeFR/parameters/3-update_calib_fr/bridge_matrices.xlsx`. When an extension exists in both sets of accounts, the pipeline keeps that of the France calibration.

The France calibration carries version number 3.10.2 and the new extensions version number 3.11.2: when `-Version` changes, the script only changes the `newer_accounts` path.

#### Step 3 — France engine

`run_engine_batch.ps1` runs the `eeio` engine with the `exo_invest_matrix` strategy, which endogenizes capital: gross fixed capital formation is no longer a final demand, it is allocated to the products that use it. With this strategy, only `A`, `Y` and `Y_k` can be shocked on the system side.

The script runs:

1. the two base years, without shock;
2. one run for each accounts folder of `Outputs/France/` (except 2019) combined with each scenario of `2-shocks/1-transitions_2050/`.

The engine applies each shock relatively, `W = W0 × (1 + dW)`, then recomputes the accounts.

The outputs are stored by **group**, according to the accounts used:

| France accounts | Output folder in `Module_PlaneFR/data/<v>/` |
|---|---|
| `Base_year_2019`, no shock | `2019_FR/` |
| `Base_year_2015`, no shock | `no_SSP/Base_year_2015/` |
| `Base_year_2015` × scenario | `no_SSP/<scenario>/` |
| `<SSP>_2050` × scenario | `<SSP>_2050/<scenario>/` |

The 2015 base year is copied into each group, where it serves as the reference of the figures.

#### Step 4 — direct pressures of final demand

The direct pressures of final demand (`F_Y`: household water use, artificial surfaces…) are not part of the data model produced by the adapter, which only copies `F` and the units. `extract_f_y_tot.py` therefore replays the loading steps of the adapter and extracts `F_Y` for each extension:

- for a pass-through extension, it reads `F_Y` in the raw satellite account;
- for a custom extension, it substitutes `F_Y` for `F` before calling the extractor, so that `F_Y` undergoes the same filtering and conversions as `F`.

The result is summed over the final demand categories to keep one column per region (`F_Y_tot`).

#### Step 5 — World, Europe and regional aggregates

`reformat_d_cba_monde_europe.py` reads `d_cba`, `F_x_dom` and `F_Y_tot` of each extension in the world accounts, then:

- for 2019, it sums the columns over all regions (`2019_W`), over the first 27 regions of the classification, which are the EU27 member states (`2019_EU27`), and over each region other than France (`2019_<code>`). The final demand category "Exports: Total (fob)" is excluded. These folders are deleted and rebuilt at each run;
- for France, it only writes `F_Y_tot`, into `dom_<ext>/`, because `d_cba` and `F_x_dom` come from the France engine. The `F_Y_tot` of each base year is copied into every France scenario folder **with the same base year**, read from the folder's `system/info.json`, wherever the folder sits: `2019_FR` and its copies in the `Base_year/` and `S1+/` groups get the 2019 value, all scenarios based on 2015 get the 2015 value. The `ghg_emissions` and `raw_materials` extensions are excluded, their direct pressures coming from the France calibration.

France scenario folders are recognized by their `dom_`/`imp_` extension folders and are never deleted. This script resets `F_Y_tot` to its base-year value: `compute_ds_x_dom.py` must run after it to reapply the `F_Y_tot` shocks.

#### Step 6 — harmonizing the GHG and materials extensions

The three extensions coming from the France calibration do not have the same format as the others. `reformat_d_cba_k.py` aligns them, so that `planefr_lib` handles all extensions in the same way. Each treatment is idempotent.

| Extension | Treatment |
|---|---|
| `dom_ghg_emissions` | Removes the `ghg_combustion` part, handled separately, then renames the `gas` index level to `indicator`. |
| `imp_ghg_emissions` | Sums the sources by gas. |
| `dom_ghg_combustion` | Collapses to a single "CO2" row. Creates `imp_ghg_combustion` filled with zeros, combustion emissions being counted as domestic. |
| `*_raw_materials` | Keeps one row per material and adds an "RMC" row, the sum of all materials. |
| `F_x_dom` of `ghg_combustion` and `raw_materials` | Missing upstream: recomputed as the sum of the domestic parts of `F_Y` and `F_Z`. |
| `M`, `M_k`, `M_RoW`, `S_x_dom`, `S_Y`, `S_Z` | Rows reformatted like those of `d_cba` or `F_x_dom`. |

#### Steps 7 and 8

They are used to build the manual shocks and are described in § 6.1.

### 5.3 SSP chain: `run_all_SSPs.ps1`

This chain first projects the rest of the world to 2050, then redoes steps 1 to 3 on these projected world accounts.

| Step | Folder | Script | Processing |
|---|---|---|---|
| 1 | `b-ssp_shocks` | `extract_S_SSPs.py` | Computes the 2050/2020 changes in pressures, activities and intensities in the IMAGE scenarios. |
| 2 | `b-ssp_shocks` | `generate_shock_dS.py` | Turns these changes into MatMat shocks on the intensities (`dS_x_dom`) and on the technical coefficients (`dA`). |
| 3 | `4-engine` | `run_engine_batch_SSPs.ps1` | World `eeio` engine, one run per SSP scenario. Output: `Outputs/World/<SSP>_2050/`. |
| 4 | `2-gmrio_to_snac_s` | `run_gmrio_to_snac_s_batch_SSPs.ps1` | `gmrio_to_snac_s` on each projected world. Output: `Outputs_inter/gmrio_to_snac_s/SSPs/<SSP>_2050/`. |
| 5 | `3-update_calib_fr` | `run_update_calib_fr_batch_SSPs.ps1` | `update_calib_fr`. Output: `Outputs/France/<SSP>_2050/`. |
| 6 | `4-engine` | `run_engine_batch.ps1` | France engine: French scenarios on each set of accounts. |
| 7 to 10 | `reformat_d_cba_monde_europe.py`, `reformat_d_cba_k.py`, `export_dS.py`, `compute_ds_x_dom.py` | As in the base chain. |

Points specific to this chain:

- **Single source for the base year of the SSP shocks.** The world accounts that the engine shocks are those of `path_in.accounts` in `settings_engine_SSPs.json`, which `run_engine_batch_SSPs.ps1` leaves unchanged (currently `Outputs\World\Base_year_2015`). `generate_shock_dS.py` and `generate_shock_agri.py` read the structure of the shocks (`S_x_dom`, `A`, `detail_levels.xlsx`, `base_year` of the `info.json` files) in the same folder, through `planefr_paths.get_ssp_base_accounts_dir()`. To change the base year, edit that key and rerun `run_all_SSPs.ps1`.
- **French domestic pressures are not changed by the SSP scenario.** At step 4, the `F_x_dom` of each scenario are replaced by those of the reference output `SSPs/Base_year_2015`. The SSP scenario therefore only acts on the pressure content of imports. The change in domestic pressures belongs to the French scenario. The reference folder is not created by any script of this chain: it must already contain the `gmrio_to_snac_s` output of the 2015 base year.
- **`extract_f_y_tot.py` is not rerun.** It only depends on raw EXIOBASE and must have been run once by the base chain.

### 5.4 Agri-food chain: `run_all_SSPs_agri.ps1`

This variant produces results in which only the pressures of the agri-food system remain. It does not change the production system: it cancels the pressure intensities of the other sectors with a `dS = −1` shock.

`generate_shock_agri.py` creates the `1-transitions_2050_agri` and `3-SSPs_2050_agri` shock sets, plus a `Base_year_agri` scenario without system shock. The batch scripts are then called with `-Suffix _agri`, which appends this suffix to every folder read and written. The construction of the shocks is described in § 6.3.

The groups produced are `no_SSP_agri/` and `<SSP>_2050_agri/`, with `Base_year_agri` as reference. The last step runs `compute_ds_x_dom.py --only-f-y-tot`: step 6 of the chain resets `F_Y_tot` in every scenario folder, `_agri` or not, so the `F_Y_tot` shocks are reapplied without recomputing the `dS_x_dom`.

### 5.5 Assembling the displayed groups

The `ADEME`, `PAPER`, `S1+` and `Base_year` groups combine or rename scenarios computed in other groups. They are declared in [data/scenario_groups.json](../data/scenario_groups.json), which maps each displayed folder to its source, an engine output relative to `data/<version>/`:

```json
"PAPER": {
  "2015": "no_SSP/Base_year_2015",
  "S1": "no_SSP/S1_2050",
  "S1+": "SSP2_19_Sus_2050/S1_2050",
  "S4": "no_SSP/S4_2050",
  "TREND": "no_SSP/TEND_2050"
}
```

[build_scenario_groups.py](../../MatMat/matmat-ademe/planeFR/scripts/6-scenario_groups/build_scenario_groups.py) rebuilds them at the end of `run_pipeline_from_engine.ps1`, `run_full_pipeline.ps1` and `run_all_SSPs.ps1`, after `compute_ds_x_dom.py`, so that the copies carry the final `F_Y_tot` values. Each declared folder is deleted then recopied from its source; undeclared folders of a group are left untouched and reported. The script stops with exit code 1, without copying anything, if a source is missing or if a group is the parent folder of a source (the engine output groups are never overwritten). It can also be run alone, e.g. `python build_scenario_groups.py --groups S1+`.

The group folders must therefore not be edited by hand. To add or change a display group, edit `scenario_groups.json` and rerun the script.

| Group | Folder | France accounts | Shock |
|---|---|---|---|
| `ADEME` | `2015` | `Base_year_2015` | none |
| | `S1` to `S4`, `TREND` | `Base_year_2015` | `S1_2050` to `S4_2050`, `TEND_2050` |
| `PAPER` | `2015` | `Base_year_2015` | none |
| | `S1`, `S4`, `TREND` | `Base_year_2015` | `S1_2050`, `S4_2050`, `TEND_2050` |
| | `S1+` | `SSP2_19_Sus_2050` | `S1_2050` |
| `S1+` | `2019_FR` | `Base_year_2019` | none |
| | `S1_2050` | `SSP2_19_Sus_2050` | `S1_2050` |
| `Base_year` | `2019_FR` | `Base_year_2019` | none |

The exact origin of a folder can be read in its `metadata.json`, written by the engine, which records the paths of the accounts and shock used. Its `system/info.json` gives its base year and scenario name; `reformat_d_cba_monde_europe.py` and `compute_ds_x_dom.py` rely on it, so copied folders receive the right `F_Y_tot` values and shocks.

`export_dS.py` reads the `no_SSP` engine outputs under their explicit names (`Base_year_2015` and the `SCENARIO_FOLDER_NAMES` of `compute_ds_x_dom.py`), and stops with an explicit message if one is missing. Other folders placed in `no_SSP/` are ignored.

> **Point of attention.** At the time of writing, the folders of `no_SSP/` have been renamed by hand (`Base_year`, `S1`… instead of `Base_year_2015`, `S1_2050`…). The `no_SSP` sources of `ADEME` and `PAPER` therefore do not exist under their engine names until the France engine is rerun, which recreates them next to the renamed folders. These renamed folders then become redundant duplicates in the `no_SSP` group, which is the default group of `planefr_lib` (`PLANEFR_SCENARIO_GROUP`).

---

## 6. Building the scenario shocks

A MatMat shock is a folder with the same structure as accounts, in which each variable `dW` gives the relative change of `W`.

| Shock set | Origin | Variables |
|---|---|---|
| `1-transitions_2050/<S>/system/` | Delivered by ADEME (MatMat shock adapter, not reproduced here) | `dA`, `dY`, `dY_k` |
| `1-transitions_2050/<S>/extensions/dom_<ext>/` | `compute_ds_x_dom.py` | `dS_x_dom` |
| `3-SSPs_2050/<SSP>_2050/system/` | `generate_shock_dS.py` | `dA` |
| `3-SSPs_2050/<SSP>_2050/extensions/<ext>/` | `generate_shock_dS.py` | `dS_x_dom` |
| `_agri` sets | `generate_shock_agri.py` | `dS_x_dom`, `dS_Z`, and copy of `dA`, `dY`, `dY_k` |

### 6.1 Manual shocks on French domestic pressures

The delivered Transition(s) 2050 scenarios only cover the production system. For the extensions added by PlanéFR, the change in domestic pressures to 2050 is entered by hand in `planeFR/parameters/a-manual_shocks/my_manual_param.xlsx`.

This workbook has one sheet per extension (`dom_air_emissions`, `dom_land`, `dom_nutrients`, `dom_water`). Each row designates a pressure (`row_name`) and gives, for each scenario, its relative change between 2015 and 2050. The `variable` column is `F_x_dom` for a production pressure or `F_Y_tot` for a direct pressure of final demand. The values themselves are scenario assumptions (source to be specified).

**Production pressures.** The shock entered applies to the total pressure `F`. The engine, however, shocks the intensity `S = F / x` and recomputes `F = S × x`. Since production `x` also changes between 2015 and 2050, applying the entered shock directly to `S` would not give the targeted pressure. `compute_ds_x_dom.py` therefore computes the intensity shock that does:

```python
f_row_2050 = f_row_base * (1 + shock)                 # targeted pressure in 2050
s_base     = f_row_base / x_dom.loc["Base_year_2015"]
s_2050     = f_row_2050 / x_dom.loc[folder_name]      # 2050 production of the scenario
ds_row     = s_2050 / s_base - 1
```

The shock is applied uniformly to all the sectors exerting the pressure. An empty or zero shock leaves `dS = 0`: it means "not calibrated", not "constant pressure". In that case the intensity is unchanged and the pressure follows production.

The base-year data and the 2050 production are read in `Module_PlaneFR/data/S_extraction.xlsx`, written by `export_dS.py` from the engine outputs: an `x` sheet with one row per scenario, and one `Fx_<ext>` and one `FY_<ext>` sheet per extension.

**Direct pressures of final demand.** They have no sectoral dimension and do not go through the engine. The script writes the base-year value multiplied by `(1 + shock)` directly into the scenario's `F_Y_tot.pkl`. Starting from the base-year value, not from the value already in the file, makes the operation idempotent.

These shocks are applied to **every** scenario folder of `data/<version>/`, whatever its group (`no_SSP`, SSP groups, `_agri` groups, groups assembled by hand). The scenario of a folder is read from its `system/info.json` (`scenario_name` = S1…TEND and `base_year` = 2015), not from its name: `PAPER/S1+`, for example, receives the S1 shock. Base years without shock (empty `scenario_name`, `Base_year_agri`) are left out.

The `--only-f-y-tot` option applies the `F_Y_tot` shocks without recomputing or rewriting the `dS_x_dom`.

### 6.2 Shocks on the rest of the world from the IMAGE scenarios

#### Principle

EXIOBASE provides the base-year intensities. The IMAGE scenarios provide, by large region, trajectories of pressures and activities. The module only takes **relative changes** between 2020 and 2050 from IMAGE, which it applies to the EXIOBASE intensities:

```
S_2050 = S_base × (F_2050 / F_2020) / (x_2050 / x_2020)
```

`F` is a pressure and `x` the activity generating it, both read in IMAGE. The ratio of the two changes gives the change in intensity.

#### The matching workbook

`Module_PlaneFR/data/SSP/Matching_extensions_EXIOBASE_IMAGE.xlsx` holds all the parameters.

| Sheet | Content |
|---|---|
| `Matching` | One row per rule: group (`EXIOBASE groups`), target extension row (`EXIOBASE stressors`), sectors concerned (`EXIOBASE products`), IMAGE formula of the numerator (`F (IMAGE)`), IMAGE formula of the denominator (`x (IMAGE)`), `Remark`. Filled in by hand. |
| `Regions` | Mapping between IMAGE regions, IMAGE-GNM regions and 19 aggregated regions plus the World (`Code aggregated` column). |
| `SSP2`, `SSP2_19`, `SSP2_19_Sus` | Results by aggregated region (F, x and S 2050/2020). Sheets **generated** by `extract_S_SSPs.py`. |

`extract_S_SSPs.py` renames the columns of the `Matching` sheet by position, so a change of spelling in the headers does not break it.

The `Matching` sheet contains 143 rules, grouped as follows:

| Group | Rules | Variable shocked |
|---|---|---|
| `Air_emissions`, `Nutrients`, `Land`, `Water`, `Raw_materials`, `Ghg_emissions` | 94 | Intensities `S` of the extension of the same name |
| `Cultures → élevage` | 1 | Coefficients `A`: plant feed per unit of animal product |
| `Energy → <group>` (8 user groups: Steel, Chemicals, Non-Metallic Minerals, Paper, Other Industries, Commercial, Transportation, Agriculture) | 34 | Coefficients `A`: energy use by carrier |
| `Electricity → All` | 10 | Coefficients `A`: shares of the power generation technologies |
| `Primary fossil → Electricity` | 4 | Coefficients `A`: fuel per unit of electricity |

An arrow "→" in the group column designates a shock on matrix `A`. Any other value designates a shock on the intensities `S` of the extension whose name is the group name in lower case.

The labels of the `A` groups are matched literally by `generate_shock_dS.py` (constants `ENERGY_GROUP_PREFIX`, `ELECTRICITY_MIX_GROUP`, `HEAVY_GROUPS`): renaming a group in the workbook requires changing these constants. If the scenario sheets contain a group missing from the `Matching` sheet, the script stops and asks for `extract_S_SSPs.py` to be rerun.

Region mapping goes through two tables: the `Regions` sheet links the IMAGE regions to the aggregated regions, and the `regions` sheet of `planeFR/parameters/4-engine/bridge_SSPs.xlsx` links each of the 49 EXIOBASE regions to an aggregated region. All the EXIOBASE regions of the same aggregated region get the same shock.

#### Step 1 — `extract_S_SSPs.py`: 2050/2020 changes

For each file of `data/SSP/Scenarios/*.csv` and each aggregated region, the script evaluates the formulas of the `Matching` sheet in 2020 and in 2050, then computes the three ratios.

The formulas are written in plain text in the cells. A parser interprets them with the following rules:

| Element | Rule |
|---|---|
| Operators | `+`, `-`, `x`, `/`, surrounded by spaces; numbers with a decimal comma |
| Parentheses | A parenthesis placed after a variable name is a unit and is ignored. The others group. |
| `max (A, B)`, `min (A, B)` | Applied year by year, e.g. for a floor at zero. |
| `A OR B` (also written `A OU B`) | Equals `A`, unless `A` is negative in 2020 or in 2050: `B` is then used. |
| `1 / <formula>` | Designates a yield: the computed ratio is directly that of the intensity. |

The variables are read in the scenario's CSV file. For the `Nutrients` group, they are read first in `Nutrients.xlsx` (IMAGE-GNM outputs, scenario `PB_<scenario>`).

Computation assumptions:

- **Common starting year.** The 2020 values of all scenarios are those of `SSP2`. Only the 2050 values are specific to each scenario.
- **Regional aggregation.** The values are summed by aggregated region before computing the ratios. A missing or zero World value is replaced by the sum of the regions.
- **Intensity that cannot be computed.** If `F` or `x` is missing, the intensity is left unchanged (`S = 1`).
- **Negligible intensity.** If the 2020 intensity of a region is below 1 % of the world intensity (`MIN_RELATIVE_INTENSITY`), it is left unchanged. The 2050/2020 ratio of an almost-zero value makes no sense and can reach orders of magnitude that make the Leontief matrix unusable.
- **Missing yield.** If a `Yield|...` yield is missing from a scenario, the rule takes the variables of the "Cropland" row.

The scenario sheets are deleted and recreated at each run. The script also writes, in the `Matching` sheet, the world change in intensity for each scenario (`S 2050/2020 (World <scenario>)` columns) and the world F and x of SSP2.

#### Step 2a — `generate_shock_dS.py`: shocks on intensities

For each scenario and each extension, the script builds `dS = S_2050/S_2020 − 1`, in the format of the `S_x_dom` of the base accounts (§ 5.3). The cells not covered by a rule stay at zero.

A rule is attached to the rows of the extension as follows:

1. if the label of the rule is exactly that of a row, only that row is shocked;
2. otherwise the label is split on " - " and every row containing each of the segments is shocked. "Cropland - Wheat" thus targets "Cropland - cropped area - Wheat" and "Cropland - fallowed area - Wheat";
3. if several rules cover the same cell, the most specific one wins, i.e. the one targeting the fewest rows.

The shock only applies to the sectors listed by the rule whose intensity is non-zero in at least one region of the base accounts.

#### Step 2b — `generate_shock_dS.py`: shocks on technical coefficients

Switching from one energy source to another, or from one power technology to another, does not belong to the intensities: it changes the purchases between industries, hence matrix `A`. The script computes a matrix `A'` from the `A` of the base accounts, in money terms, in three steps.

1. **Single factor.** For `Cultures → élevage`, `Primary fossil → Electricity` and `Energy → Agriculture`, the coefficients are multiplied by the change in intensity: `A' = A × S_2050/S_2020`.
2. **Energy user groups.** Each group (steel, chemicals, commercial…) has one row per energy carrier.
   - *Intensity.* The total energy purchased by the group is multiplied by `I`, the ratio of the IMAGE energy intensity in 2050 to that of 2020. `I` is 1 if it cannot be computed or if the 2020 intensity of the region is negligible.
   - *Shares.* The share of each carrier is shifted by the change in its IMAGE share between 2020 and 2050, then the shares are renormalized.
   - For steel, chemicals and non-metallic minerals (`HEAVY_GROUPS`), only the intensity is shocked and the EXIOBASE shares are kept.
3. **Electricity mix.** The same share computation is applied to the power generation technologies, at constant total electricity.

Within a carrier or technology, the coefficients are scaled proportionally: the weights of the products and of the regions of origin are kept.

The shock is then expressed in relative form, `dA = A'/A − 1` where `A > 0` and 0 elsewhere. This has a consequence: a zero coefficient in `A` stays zero. A carrier missing from a column therefore cannot appear in it. For `A'` to be reached exactly, the shares are renormalized over the carriers present only, i.e. those with a share above 1 % (`MIN_VECTOR_SHARE`).

**Safeguard.** Before writing, the script computes the spectral radius of `A'`. The Leontief inverse is finite and positive only if this radius is below 1. If it reaches 0.95 (`MAX_SPECTRAL_RADIUS`), no shock on `A` is written for the scenario and an earlier `dA.pkl` is deleted. Columns whose sum exceeds 1 while increasing, which corresponds to a negative value added, are only reported in the console output.

`dA` is only written as a pickle: the matrix has 9,800 rows and 9,800 columns.

### 6.3 Agri-food shocks

`generate_shock_agri.py` derives each `_agri` scenario from the original scenario. It copies the system shocks unchanged, then rewrites the intensity shocks:

- for the sectors kept, `dS` keeps the value of the original scenario, or 0 if the scenario does not shock the extension;
- for all other sectors, `dS = −1`, which cancels the intensity.

The sectors kept are those marked 1 in the `Food` column of the `M` sheet of `Module_PlaneFR/data/bridge_matrices.xlsx`, plus "Products of forestry, logging and related services (02)" and minus "Hotel and restaurant services (55)". For the SSP shock sets, the base accounts are the same as in § 5.3.

| Extension | Variable shocked | Filter |
|---|---|---|
| General case | `dS_x_dom` | Columns of the sectors kept |
| `dom_ghg_emissions`, `ghg_emissions` | `dS_x_dom` | Sectors kept and fertilizers ("N-fertiliser", "P- and other fertiliser") |
| `dom_ghg_combustion` | `dS_Z` | Columns of the sectors kept and of the fertilizers |
| `dom_raw_materials` | `dS_Z` | Rows of the sectors kept: all the biomass extracted, whatever the sector using it |

The direct pressures of final demand (`S_Y`) are not shocked.

---

## 7. Post-processing and figures: Module_PlaneFR

### 7.1 Structure of the model results

`Module_PlaneFR/data/<version>/` contains one folder per scenario, either directly or in a group folder.

A France scenario has its extensions split into `dom_<ext>` and `imp_<ext>`:

```
<scenario>/
├── metadata.json                     parameters of the engine run
├── system/                           A, K, L, L_k, Y, Y_k, Z, x, unit, detail_levels.xlsx, info.json
└── extensions/
    ├── dom_land/                     F_x_dom, S_x_dom, F_Y_tot, M, M_k, d_cba, d_cba_k, unit
    └── imp_land/                     M_RoW, d_imp, M, M_k, d_cba, d_cba_k
```

A World, Europe or region folder has a single folder per extension, with three already aggregated files: `d_cba.pkl`, `F_x_dom.pkl` and `F_Y_tot.pkl`.

`planefr_lib` tells the two cases apart by the folder structure, not by its name (`processing.scenario_has_dom_imp_split`).

### 7.2 The `planefr_lib` library

The computation and plotting code is gathered in [notebooks/planefr_lib/](../notebooks/planefr_lib/). The notebooks only contain the parameters and the calls.

| Module | Role |
|---|---|
| [config.py](../notebooks/planefr_lib/config.py) | Paths, scenario groups, sheet and column names of the parameter workbooks, row names of the thresholds, share columns, colors |
| [io.py](../notebooks/planefr_lib/io.py) | Reading the parameter workbooks and the model pickles |
| [processing.py](../notebooks/planefr_lib/processing.py) | Footprint and budget computations |
| [colors.py](../notebooks/planefr_lib/colors.py) | Colors by consumption category and by scenario |
| [plot_stacked_bar.py](../notebooks/planefr_lib/plot_stacked_bar.py) | Stacked bars and synthesis figures |
| [plot_radial_synthesis.py](../notebooks/planefr_lib/plot_radial_synthesis.py) | Radial wheel of the planetary boundaries |
| [plot_overshoot.py](../notebooks/planefr_lib/plot_overshoot.py) | Overshoot of the safe operating space, by region |
| [plot_bubble.py](../notebooks/planefr_lib/plot_bubble.py) | Intensities by product or by category, and the associated Excel export |
| [plot_process_breakdown.py](../notebooks/planefr_lib/plot_process_breakdown.py) | Contribution of each extension to a sub-process |

Main loaders of `io.py`:

| Function | Reads |
|---|---|
| `load_characterization_factors(scenario_group)` | `characterization_factors.xlsx`, sheet `All`, or `Agri` for a group ending in `_agri`. Column names are matched case-insensitively. |
| `load_thresholds(scenario_group)` | `thresholds.xlsx`, sheet `All` or `Agri`. Row and column labels are normalized (repeated spaces collapsed). |
| `load_dls()` | `thresholds.xlsx`, sheet `DLS` |
| `load_bridge_matrices()`, `load_bridge_m_matrix(sheet)` | `bridge_matrices.xlsx` |
| `load_budget_shares_df()` | `budget_shares.xlsx`, sheet `Budget_shares_regions`, plus the `UN – EPC <period>` columns of `Budget_shares_variants` |
| `load_population_df()` | `budget_shares.xlsx`, sheet `Population`, indexed by EXIOBASE region code, one column per period |
| `load_bdf(sheet, scenario_group)` | Household workbooks of `ademe_eemrio` (§ 8) |

The pickle loaders return `None` when a file does not exist for a scenario × extension pair. This is a normal case: `ghg_combustion`, for example, does not exist in the World and Europe folders.

**Selecting scenarios.** `config.get_scenario_folders(exclude_2019, exclude_others, scenario_group)` returns the folders of a group, sorted by name. A folder is recognized as a scenario if it contains an `extensions` sub-folder. The reference scenario of the figures is the first in the list (`REFERENCE_SCENARIO_IDX = 0`). If the group has no `Base_year_2015` folder, a warning is issued and the reference is the first folder in alphabetical order: `2015` in `ADEME` and `PAPER`, `2019_FR` in `S1+`.

> **Possible improvement — library structure.**
> - `planefr_lib` is imported by adding `notebooks/` to `sys.path`, both in the notebooks and in `extract_S_SSPs.py`; making it an installable package (`pip install -e`) would remove these path manipulations.
> - `plot_overshoot.py` and `plot_radial_synthesis.py` exceed 1,000 lines each; splitting geometry, scales and annotations into separate modules would make them easier to maintain.

### 7.3 Computing a footprint

#### From extensions to sub-processes

`data/characterization_factors.xlsx` makes the link. Its `All` sheet lists 653 rows with five columns: `Indicators` (sub-process), `EXIOBASE groups` (extension), `EXIOBASE stressors` (row of the extension), `Units`, `Characterization factors`. The `Agri` sheet (126 rows) is read instead for groups whose name ends in `_agri`.

The pressure of a sub-process is the sum, over its rows, of the value of the row multiplied by its factor:

```
pressure(sub-process) = Σ_rows  factor(row) × value(row)
```

> **Possible improvement — performance.** `processing.filter_and_weight` applies this formula row by row with `iterrows`. A vectorized version (aligning the factors on the index with `reindex`, then one multiplication) would give the same result much faster, which matters with 653 factor rows (408 for "Biodiversity loss - PDF" alone) applied to every scenario.

| Sub-process | Rows used | Factor |
|---|---|---|
| GHG emissions | 6 gases of `ghg_emissions`, CO2 row of `ghg_combustion` | 1 (MtCO2eq) |
| N surplus | N - agriculture - water; NH3 and N2O - agriculture - air | 1e-9; 8.24e-10; 6.36e-10 |
| P surplus | P - agriculture - water and - soil | 3e-10 |
| Blue water consumption | 129 "Water Consumption Blue" rows | 1 (Mm³) |
| Cropland use | 21 cropland rows | 1 (km²) |
| Agricultural land | 24 cropland and pasture rows | 1 (km²) |
| Raw material consumption | RMC row | 1 (kt) |
| Biodiversity loss - PDF | 408 rows from six extensions | PDF.m2.yr per unit of pressure |
| Biodiversity loss - HANPP | 24 rows of `land`, 34 of `raw_materials` | 5.3e-8 to 5.5e-7 |

The NH3 and N2O factors correspond to the nitrogen mass fraction of each molecule (14/17 and 28/44), multiplied by 1e-9 to go from kilograms to teragrams. The origin of the other factors is not documented in the repository (source to be specified). The factors of "Biodiversity loss - PDF" were added by `data/add_biodiversity_rows.py` from IMPACT World+ 2.2.1, by summing the indicators in PDF.m2.yr, excluding the "long term" indicators. This is a one-shot script: the IMPACT World+ file is not kept in `data/` and must be put back next to the script to rerun it.

#### Consumption footprint of a France scenario

`processing.process_scenario` computes, for each sub-process:

1. **Reading.** `d_cba_k.pkl` of `dom_<ext>` and `imp_<ext>`. This is the footprint computed with endogenized capital: one row per pressure, one column per product × final demand category pair (200 products × 5 categories).
2. **Characterization.** Filtering the rows of the sub-process and weighting them by their factor (`filter_and_weight_vector`).
3. **Breakdown.** Projection of the 1,000 columns on five consumption categories with the `Final conso. cat.` sheet of `bridge_matrices.xlsx`: Food, Housing, Mobility, Final goods, Final services. The export columns have zero weight: the footprint is that of households and general government.
4. **Conversion.** Division by the "Unit conversion Exiobase" row of `thresholds.xlsx`, to go from the EXIOBASE unit to the figure unit.
5. **Final demand.** Addition of `F_Y_tot`, if it exists, in a separate category called `F_Y` ("Not allocated" in the legends), counted as domestic.
6. **Aggregation.** Sum over the extensions of the sub-process.

The result is a payload with the keys `"domestic"` and `"imported"` (constants `processing.DOMESTIC` and `processing.IMPORTED`), each a series by consumption category, plus `"pba"` and `"categories"`.

#### Territorial pressures

The PBA value of a sub-process is the characterized sum of `F_x_dom` and `F_Y_tot`, converted in the same way. It is a single total, with no breakdown by category.

#### Variants

| Function | Output | Scenarios accepted |
|---|---|---|
| `process_scenario` | CBA by category and origin, PBA total, absolute values | France |
| `process_scenario_absolute` | CBA and PBA totals, absolute values | France, World, Europe, regions |
| `process_scenario_per_capita` | CBA and PBA totals per capita | France, World, Europe, regions |
| `process_subprocess_lp_breakdown` | Footprint of a sub-process by extension of origin | France, World, Europe |

Only `process_scenario` reads `d_cba_k.pkl`. The three other functions read `d_cba.pkl`, in which capital formation remains a final demand category, and remove the "Exports" category from it.

For the per-capita computation, the population is read in the `Population` sheet of `budget_shares.xlsx` (UN World Population Prospects 2024, thousands of inhabitants). The period is deduced from the folder name (`2019`, `2015`, otherwise `2050`), and the region from the region code in the name (`processing.get_population`).

### 7.4 Thresholds and budgets

#### Global budgets

The `All` sheet of `data/thresholds.xlsx` gives, for each sub-process, three levels of annual global budget. Values at the time of writing (source to be specified; the two GHG columns are computed by formulas in the cells):

| Sub-process | Lower safe bound | Safe limit | Upper safe bound | Unit |
|---|---|---|---|---|
| GHG emissions | 2.39 | 4.55 | 10.84 | GtCO2eq |
| GHG emissions (2019) | 6.15 | 9.74 | 16.93 | GtCO2eq |
| Blue water consumption | 1,100 | 2,800 | 4,500 | km³ |
| Cropland use | 10.6 | 12.6 | 14.6 | M km² |
| Agricultural land | — | 43.2 | 48 | M km² |
| N surplus | 34 | 57 | 74 | Tg |
| P surplus | — | 4.5 | 9 | Tg |
| Raw material consumption | — | 25 | 50 | Gt |
| Biodiversity loss - PDF | — | 6.5 | 45.5 | 10¹² PDF.m2.yr |
| Biodiversity loss - HANPP | — | 5.5 | 11 | GtC |

The `Agri` sheet only has a "Safe limit" row, for seven sub-processes (GHG emissions: 5 GtCO2eq; Blue water consumption: 2,000 km³; the others as above).

The same sheet contains the unit-conversion rows:

| Row | Use |
|---|---|
| `Unit conversion budget` | Divisor: from the unit of the global budget to the figure unit |
| `Unit conversion Exiobase` | Divisor: from the EXIOBASE unit to the figure unit |
| `Figures unit` | Unit displayed |
| `(p.cap)` variants | Same roles for per-capita values |
| `Equality…` | Precomputed national thresholds, read when no sharing principle is requested |

The `Equality…` rows are formulas linked to `budget_shares.xlsx`: global budget × France share of the `Budget_shares_variants` sheet (`UN – EPC 2019-2100` for GHG, `UN – EPC 2050` otherwise) for the absolute rows, and population of the `Population` sheet for the `(p.cap)` rows. These links are by cell reference: moving rows or columns in `budget_shares.xlsx`, or renaming its sheets, breaks them.

> **Point of attention.** The external links of `thresholds.xlsx` still point to a sheet called `Budget_shares_variantes`, the former name of `Budget_shares_variants` (30 cells). The values cached in the file remain correct and are what pandas reads, but Excel cannot refresh these formulas until the links are updated.

> **Possible improvement — fragile dependencies on the Excel files.** Several reads depend on the layout of the workbooks rather than on explicit keys: cell links between `thresholds.xlsx` and `budget_shares.xlsx` (broken as soon as a sheet is renamed, as above, or a row or column moved); fixed positions (`iloc[1:201]` in `io.load_bridge_m_matrix`, `header=3` for `budget_shares.xlsx`, `iloc[0, 7:]` in `compute_ds_x_dom.load_manual_shocks`); scenario properties deduced from folder names (`"2019" in name` in `processing.get_population`). Computing the national thresholds in `processing` from the global budgets and the shares, instead of reading formula results, and storing the parameters as long-format tables with explicit key columns, would remove these failure modes.

#### From global to national budget

The `sharing_principle` parameter of the plotting functions chooses the rule. Allowed values are `None`, `"EPC"` and `"CTR"`.

| Value | Budget of France (or of a region) |
|---|---|
| `None` | Direct reading of the `Equality…` rows of `thresholds.xlsx` |
| `"EPC"`, `"CTR"` | `Global budget / conversion × share of the region`, share read in `budget_shares.xlsx` (`processing.compute_region_budget`) |

With `"EPC"`, the share column depends on the figure and on the sub-process:

| Figure | Default column | Exception for GHG emissions |
|---|---|---|
| Radial wheel, synthesis by sub-process | `UN – EPC 2050` | `UN – EPC 2019-2100` |
| Overshoot by region, stacked bars by scenario | `UN – EPC 2019` | `UN – EPC 2019-2100`; budgets read in the "GHG emissions (2019)" column for the by-region figure, and for the stacked bar of a 2019 scenario |

With `"CTR"`, the reference column is `CTR share ref. (SSP2, Pop 2019 × GDP 1990-2019)`. The sensitivity bounds of the overshoot figure are the `<principle> min` and `<principle> max` columns.

These choices are set in `config.py` (`RADIAL_EPC_PATTERN`, `REGION_FIGURE_EPC_REF_PATTERN`, `REGION_FIGURE_THRESHOLD_COLUMN_BY_SUBPROCESS` and their variants by sub-process). The `plot_stacked_bar` functions choose between the two sets with their `epc_reference` parameter, `"radial"` or `"region"`.

#### Budget shares by region

`data/budget_shares.xlsx` is produced by [data/budget_shares.py](../data/budget_shares.py). It gives the share of the global budget of each of the 49 EXIOBASE regions and of the EU27, under two principles:

```
EPC_s(P)    = Pop_s(P) / Σ_s Pop_s(P)
CTR_s(P, G) = [Pop_s(P) / GDPcap_s(G)] / Σ_s [Pop_s(P) / GDPcap_s(G)]
```

EPC (*equal per capita*) allocates the budget in proportion to population. CTR (*capability to reduce*) allocates it in inverse proportion to GDP per capita. `P` is a population period and `G` a GDP period. The script computes 3 EPC and 18 CTR variants for each of the five SSPs, plus three EPC variants based on the UN population.

| Sheet | Content |
|---|---|
| `Budget_shares_regions` | Reference shares (`EPC share ref.`, `CTR share ref.`), min, mean and max over the variants |
| `Budget_shares_variants` | One column per variant, including `UN – EPC 2019`, `UN – EPC 2050`, `UN – EPC 2019-2100` |
| `Population` | UN population by region, for 2019, 2050, 2019-2100, 2015 and 2019-2050 (mean over the period) |
| `Country_diagnostics` | Detail by economy: region, inclusion, projection used |

The shares are computed country by country, then summed by region. Historical data (1990-2025) come from the World Bank, projections from the IIASA SSP database. The projections are spliced on the last observed point: the level remains that of the World Bank, only the dynamics are taken from the SSPs. The details of sources, splicing and special cases are in the header of the script.

#### Social floor

The `DLS` sheet of `thresholds.xlsx` gives a decent-living-standards floor for three sub-processes (GHG emissions, Agricultural land, Raw material consumption), per capita and nationally. The "DLS" rows are used for the base-year groups, the "DLS 2050" rows for the 2050 groups. Only the national values are read by the code; a missing level (e.g. lower or upper bound) is not drawn.

### 7.5 Notebooks and figures

Each notebook sets a scenario group in its `SCENARIO_GROUP` variable, loads the data, computes the footprints, then calls a plotting function.

| Notebook | Plotting function | Files produced in `figures/` |
|---|---|---|
| `footprint_decomposition.ipynb` | `create_stacked_bar_chart`, `create_synthesis_figure_by_lp` | `scenario_NN_<name>.png`, `synthesis_by_lp_all_scenarios.png` |
| `radial_synthesis.ipynb` | `create_radial_synthesis_figure` | `<group>/radial_synthesis_cba_all_scenarios.png`, `..._pba_...png` |
| `multi-regions_synthesis.ipynb` | `create_overshoot_safe_space_figure_by_region` | `overshoot_safe_operating_space_by_region.png` |
| `driver_comparison.ipynb` | `export_subprocess_tables_excel`, `plot_bubble_chart` | `bubble_data_by_subprocess_<scenario>.xlsx`, `bubble_chart_*.png` |
| `biodiversity_process_contribution.py` | `create_subprocess_contribution_chart` | `biodiversity_loss_process_contribution.png` |

`Untitled-1.ipynb` is a draft.

**Stacked bars and synthesis by sub-process.** One bar per scenario, stacked by consumption category, domestic part solid and imported part hatched. The synthesis in absolute values shows the three threshold levels, the social floor, the PBA value and the change relative to the reference scenario.

**Radial wheel.** One sector per sub-process, one bar per scenario. The height of a bar is the ratio of the footprint to the national "Safe limit", which therefore forms a circle of constant radius. `show_cba` and `show_pba` choose the accounting displayed. The `Bdf` parameter replaces the scenarios by household categories (§ 8).

**Overshoot of the safe operating space, by region.** One row per sub-process, one bubble per scenario, positioned in multiples of the budget. `create_overshoot_safe_space_figure_by_region` works in absolute values and gives each country or region its own budget share, deduced from the region code in the folder name (`processing.resolve_region_code`). A name without a region code is treated as France. With `sensitivity=True`, a segment spans the overshoot levels obtained with the min and max shares; for the World (share 1), it is obtained per capita over the population periods 2019, 2019-2050 and 2019-2100 (`config.WORLD_SENSITIVITY_POPULATION_PERIODS`, `processing.compute_world_period_budget`), which requires `pop_df`. The notebook currently calls it with `sharing_principle="CTR"`.

**Intensities by product.** The notebook works in two steps. It first exports one workbook per scenario, with one sheet per sub-process: multipliers `M_dom`, `M_imp`, `M_row` and final demands `Y_dom`, `Y_imp` by product and by category. It then plots the bubbles by reading these workbooks back. The display order is set by the sort row chosen in cell B1 of each sheet, editable in Excel. The multipliers of a category are an average weighted by final demand. The grouping sheet is `M` (5 categories) or `M_split` (9 categories: Vegetal products, Animal products, Sugar, Beverages & Tobacco, Housing, Mobility, Textiles, Final goods nec, Hotel & restaurant, Final services).

---

## 8. Breakdown by household category: ademe_eemrio

This chain splits the consumption footprint of a scenario between household categories, in proportion to their expenditure observed in the 2017 Budget de famille survey. The `ademe_eemrio` repository predates the project: the imputation notebook and the chart notebooks were already there. PlanéFR adds two scripts and an orchestration script. The repository's own files and output sheet names are in French.

| File | Status | Role |
|---|---|---|
| [export_impacts_coicop.py](../../Github/ademe_eemrio/export_impacts_coicop.py) | Added by PlanéFR | Footprint of the scenario by consumption item |
| [correspondance_exiobase_coicop.csv](../../Github/ademe_eemrio/correspondance_exiobase_coicop.csv) | Added by PlanéFR | Mapping between the 200 EXIOBASE products and the level-1 COICOP items |
| `imputation_impacts.ipynb` | Pre-existing | Imputation to households |
| [agregation_impacts_facteurs.py](../../Github/ademe_eemrio/agregation_impacts_facteurs.py) | Added by PlanéFR | Aggregation by household category |
| [BdF.ps1](../../Github/ademe_eemrio/BdF.ps1) | Added by PlanéFR, not tracked by git | Runs the three steps for a list of scenarios |

At the time of writing, `BdF.ps1` is not committed, and `agregation_impacts_facteurs.py`, `export_impacts_coicop.py` and `imputation_impacts.ipynb` have uncommitted local changes.

### Steps

1. **Footprint by item.** `export_impacts_coicop.py` repeats the computation of `processing.process_scenario` while keeping the detail of the 200 products, then aggregates by COICOP item. The direct pressures of final demand, which belong to no product, are allocated as a block to item C04 (Housing). The script checks that its totals equal those of the figures and writes `impacts.csv`.
2. **Imputation.** The notebook splits the footprint of each item between households in proportion to their weighted expenditure on that item. It detects the indicators from the columns of `impacts.csv`. It applies no unit conversion to the PlanéFR indicators, which are missing from its `UNITES_INDICATEURS` table: the factor is 1 and a warning is displayed. Output: `impacts_facteurs.csv` and its legend.
3. **Aggregation.** `agregation_impacts_facteurs.py` sums the impacts by category and writes a workbook with three sheets: `décile` (four income brackets: `1_3`, `4_7`, `8_9`, `10`), `unité urbaine`, `croisé` (income bracket × rural `R` or urban `U`, e.g. `47U`). Each sheet has a `weight` column, the share of the population, and a total row called `Average` (constant `TOTAL_ROW`, identical to `planefr_lib.config.BDF_TOTAL_ROW`).

`BdF.ps1` associates each scenario with the `Module_PlaneFR` group that displays it, and renames the output `impacts_facteurs_agrégé_<group>.xlsx`:

```powershell
$Scenarios = @(
    @{ Scenario = "SSP2_19_Sus_2050/S1_2050"; Group = "S1+" },
    @{ Scenario = "Base_year/2019_FR";        Group = "Base_year" }
)
```

The workbooks produced are therefore read by `planefr_lib` without manual editing.

### Reading by `planefr_lib`

`io.bdf_file(scenario_group)` looks for `impacts_facteurs_agrégé_<group>.xlsx`, or otherwise a single `impacts_facteurs_agrégé_<group>_*.xlsx` file. Called with `Bdf="décile"`, the radial wheel shows one bar per household category. Each category gets a share of the national budget equal to its demographic weight. In a two-scenario group such as `S1+` (2019_FR + S1_2050), the group's file holds the 2050 scenario, and the base year is read in the `Base_year` group's file (`io.bdf_comparison_file`).

---

## 9. Data and parameter files

### Pipeline parameters

All these files are in `matmat-ademe/planeFR/parameters/` and are versioned. The JSON files are templates (§ 5): they are edited by hand and never modified by the scripts.

| File | Role | Used by |
|---|---|---|
| `1-exiobase3_eeio/exiobase3_eeio_settings.json` | Version, year, list of extensions to extract | `run_exiobase3_eeio_batch.ps1`, `extract_f_y_tot.py` |
| `2-gmrio_to_snac_s/gmrio_to_snac_S_FR.json`, `gmrio_to_snac_S_SSPs.json` | Settings of the SNAC conversion, `foreign_global_feedback` mode | `run_gmrio_to_snac_s_batch*.ps1` |
| `2-gmrio_to_snac_s/bridge_matrices.xlsx`, `detail_levels.xlsx` | Definition of the domestic region and of the final demand categories | `gmrio_to_snac_s` (via the JSON) |
| `3-update_calib_fr/update_calib_fr_FR.json`, `update_calib_fr_SSPs.json` | Extensions taken from each set of accounts | `run_update_calib_fr_batch*.ps1` |
| `3-update_calib_fr/bridge_matrices.xlsx` | From EXIOBASE resolution to that of the France calibration | `update_calib_fr` (via the JSON) |
| `4-engine/settings_engine_World.json` | World engine, base year, `standard` strategy | `run_exiobase3_eeio_batch.ps1` |
| `4-engine/settings_engine_SSPs.json` | World engine, SSP scenarios; `path_in.accounts` sets the base accounts of the SSP shocks (§ 5.3) | `run_engine_batch_SSPs.ps1`, `generate_shock_dS.py`, `generate_shock_agri.py` |
| `4-engine/settings_engine_France_shocks.json` | France engine, `exo_invest_matrix` strategy | `run_engine_batch.ps1` |
| `4-engine/bridge_SSPs.xlsx` | Mapping of the 49 EXIOBASE regions to the aggregated IMAGE regions | `generate_shock_dS.py` |
| `a-manual_shocks/my_manual_param.xlsx` | 2015-2050 change in domestic pressures, by scenario | `compute_ds_x_dom.py` |

`a-manual_shocks/detail_levels.xlsx` and `2-gmrio_to_snac_s/doc.txt` (description of the `m_row_mode` options) are kept for reference; no script reads them.

### Post-processing parameters

All these files are in `Module_PlaneFR/data/` and are versioned.

| File | Role | Produced by |
|---|---|---|
| `characterization_factors.xlsx` | Link between extension rows and sub-processes, with factors (sheets `All`, `Agri`) | by hand |
| `thresholds.xlsx` | Global budgets, unit conversions, precomputed national thresholds (sheets `All`, `Agri`), social floors (`DLS`) | by hand, formulas linked to `budget_shares.xlsx` |
| `bridge_matrices.xlsx` | Grouping of products into consumption categories (sheets `Final conso. cat.`, `M`, `M_split`) | by hand |
| `budget_shares.xlsx` | Shares of the global budget by region, UN population | `budget_shares.py` |
| `Budget share/*.csv`, `Population ONU.xlsx` | Population and GDP, historical and projected | external sources |
| `SSP/Matching_extensions_EXIOBASE_IMAGE.xlsx` | EXIOBASE-IMAGE matching and computed changes | by hand, then `extract_S_SSPs.py` |
| `SSP/Scenarios/*.csv` | Outputs of the IMAGE scenarios | external source |
| `SSP/Nutrients.xlsx` | IMAGE-GNM outputs for nitrogen and phosphorus | external source |
| `S_extraction.xlsx` | Base-year production and pressures, 2050 production | `export_dS.py` |

Sheets of `bridge_matrices.xlsx`:

| Sheet | Read by | Content |
|---|---|---|
| `Final conso. cat.` | `io.load_bridge_matrices` | 1,000 rows (product × final demand category) to 5 consumption categories |
| `M` | `io.load_bridge_m_matrix`, `generate_shock_agri.py` | 200 products to 5 categories; the `Food` column defines the agri-food system |
| `M_split` | `io.load_bridge_m_matrix` | 200 products to 9 categories |

### Python scripts of the pipeline

| Script | Main inputs | Main outputs |
|---|---|---|
| [extract_f_y_tot.py](../../MatMat/matmat-ademe/planeFR/scripts/1-exiobase3_eeio/extract_f_y_tot.py) | EXIOBASE archives, adapter settings | `Outputs/World/Base_year_*/extensions/<ext>/F_Y_tot.pkl` |
| [reformat_d_cba_monde_europe.py](../../MatMat/matmat-ademe/planeFR/scripts/5-reformat/reformat_d_cba_monde_europe.py) | World accounts of the base years | `2019_*` folders, France `F_Y_tot` |
| [reformat_d_cba_k.py](../../MatMat/matmat-ademe/planeFR/scripts/5-reformat/reformat_d_cba_k.py) | France engine outputs | Same files, reformatted |
| [export_dS.py](../../MatMat/matmat-ademe/planeFR/scripts/a-manual_shocks/export_dS.py) | France engine outputs, `no_SSP` group | `S_extraction.xlsx` |
| [compute_ds_x_dom.py](../../MatMat/matmat-ademe/planeFR/scripts/a-manual_shocks/compute_ds_x_dom.py) | `S_extraction.xlsx`, `my_manual_param.xlsx` | `dS_x_dom` of the France shocks, shocked `F_Y_tot` |
| [rebuild_shock_extensions.py](../../MatMat/matmat-ademe/planeFR/scripts/2-gmrio_to_snac_s/rebuild_shock_extensions.py) | `gmrio_to_snac_s` outputs, shocks | Extension folders of the shocks |
| [extract_S_SSPs.py](../../MatMat/matmat-ademe/planeFR/scripts/b-ssp_shocks/extract_S_SSPs.py) | IMAGE scenarios, `Matching` and `Regions` sheets | Scenario sheets of the matching workbook |
| [generate_shock_dS.py](../../MatMat/matmat-ademe/planeFR/scripts/b-ssp_shocks/generate_shock_dS.py) | Matching workbook, base world accounts | `2-shocks/3-SSPs_2050/` |
| [generate_shock_agri.py](../../MatMat/matmat-ademe/planeFR/scripts/c-agri_shocks/generate_shock_agri.py) | Original shock sets, `M` sheet | `_agri` shock sets |
| [build_scenario_groups.py](../../MatMat/matmat-ademe/planeFR/scripts/6-scenario_groups/build_scenario_groups.py) | `scenario_groups.json`, engine outputs | Display groups (`ADEME`, `PAPER`, `S1+`, `Base_year`) |
| [planefr_paths.py](../../MatMat/matmat-ademe/planeFR/scripts/_common/planefr_paths.py) | — | Shared paths, imported by the other Python scripts |
| [planefr_common.ps1](../../MatMat/matmat-ademe/planeFR/scripts/_common/planefr_common.ps1) | Settings templates | Shared paths and template functions, dot-sourced by the PowerShell scripts; generated settings in `data/4-PlaneFR/Settings_run/` |

---

## 10. Summary of assumptions

**Data and scope**

- EXIOBASE 3.11.2, product-by-product tables, years 2015 and 2019.
- The French scenarios start from the 2015 base year. The year 2019 is used for comparisons between France, Europe, the World and the regions.
- The French production system, domestic greenhouse gas emissions and domestic raw materials come from the ADEME France calibration, numbered 3.10.2. The other extensions come from EXIOBASE 3.11.2.
- The footprint by consumption category is that of households and general government: exports are excluded and capital formation is allocated to the products consumed. The cross-region figures also exclude exports, but keep capital formation as a final demand.
- The EU27 is defined as the first 27 regions of the EXIOBASE classification.

**Imports**

- The pressure content of imports is computed on the full world production system, with the direct pressures of France set to zero.
- Combustion emissions are counted as entirely domestic in their dedicated extension.

**French scenarios**

- The shocks to the production system are those of the Transition(s) 2050 scenarios delivered by ADEME.
- The change in domestic pressures of the added extensions is entered by hand, as a change in total pressure, and applied uniformly to all the sectors concerned.
- A pressure without an entered shock keeps its 2015 intensity: it follows production.
- The direct pressures of final demand keep their base-year value, unless a shock is entered; entered shocks apply in every scenario group.

**Rest of the world**

- Only the relative 2020-2050 changes of IMAGE are used, and they are applied to the EXIOBASE base-year intensities. IMAGE activities are in physical volumes or constant dollars, those of EXIOBASE in euros.
- The year 2020 is common to all scenarios and taken from SSP2.
- A regional intensity below 1 % of the world intensity in 2020 is not changed.
- An SSP scenario does not change French domestic pressures.
- An input absent from an industry in the base year cannot appear in it in 2050.
- In steel, chemicals and non-metallic minerals, only the energy intensity changes: the EXIOBASE energy mix is kept.

**Thresholds**

- The global budget is annual and is allocated between regions in proportion to population (EPC) based on UN projections, or in inverse proportion to GDP per capita (CTR). The population year or period depends on the figure and on the sub-process (§ 7.4).
- A household category gets a share of the national budget equal to its demographic weight.

---

## 11. User guide

### Rerunning all the computations

The chains are in `matmat-ademe/planeFR/scripts/0-chains/`. They can be run from any folder; from that one, in PowerShell:

```powershell
# 1. Base years and French scenarios, rest of the world unchanged
#    (two engine passes, see § 5.2; -SinglePass to skip the second one)
.\run_full_pipeline.ps1 -Version 3.11.2

#    After a change to my_manual_param.xlsx only:
.\run_pipeline_from_engine.ps1 -Version 3.11.2

# 2. French scenarios on a projected rest of the world
.\run_all_SSPs.ps1 -Version 3.11.2

# 3. Agri-food variant
.\run_all_SSPs_agri.ps1 -Version 3.11.2
```

The script of a single step can also be run on its own from its folder, e.g. `..\4-engine\run_engine_batch.ps1 -Version 3.11.2`. The Python scripts are run from the root of `matmat-ademe` with `uv run python planeFR\scripts\<folder>\<script>.py`.

Then, from `Github/ademe_eemrio/`:

```powershell
.\BdF.ps1
```

The figures are produced by running the notebooks of `Module_PlaneFR/notebooks/`.

### Running a single step

Each MatMat step can be run on its own, from the root of `matmat-ademe`. Pass a settings file generated by a previous run (`data/4-PlaneFR/Settings_run/...`), not a template of `planeFR/parameters/`: the templates contain the `{parameters}` token, which only the PowerShell scripts replace.

```powershell
uv run python -m matmat.cli -a exiobase3_eeio -st <settings.json> --no_confirm   # adapter
uv run python -m matmat.cli -p gmrio_to_snac_s -st <settings.json> --no_confirm  # pipeline
uv run python -m matmat.cli -e eeio -st <settings.json> --no_confirm             # engine
```

and from the root of `datapack`:

```powershell
uv run python -m matmat.cli -p update_calib_fr -st <settings.json> --no_confirm
```

To reapply only the manual `F_Y_tot` shocks, from the root of `matmat-ademe`:

```powershell
uv run python planeFR\scripts\a-manual_shocks\compute_ds_x_dom.py --only-f-y-tot
```

### Using `planefr_lib` in a notebook

```python
import sys
from pathlib import Path

notebooks_dir = Path.cwd()
while not (notebooks_dir / "planefr_lib").exists() and notebooks_dir != notebooks_dir.parent:
    notebooks_dir = notebooks_dir.parent
sys.path.insert(0, str(notebooks_dir))

from planefr_lib import config, io, processing
from planefr_lib.plot_radial_synthesis import create_radial_synthesis_figure

GROUP = "PAPER"
cf_df = io.load_characterization_factors(GROUP)
thresholds_df = io.load_thresholds(GROUP)
bridge_df = io.load_bridge_matrices()

folders = config.get_scenario_folders(exclude_2019=True, scenario_group=GROUP)
data = [processing.process_scenario(f, cf_df, bridge_df, thresholds_df)[0] for f in folders]

fig, _ = create_radial_synthesis_figure(
    data, thresholds_df, [f.name for f in folders],
    show_cba=True, show_pba=False,
    sharing_principle="EPC",
    scenario_group=GROUP,
)
```

### Adapting the module

| Need | Files to change |
|---|---|
| Add an EXIOBASE extension | Add its name to `extension_names` in the templates of `planeFR/parameters/` (adapter, `gmrio_to_snac_s`, engines); add `dom_<ext>` and `imp_<ext>` to the `update_calib_fr` settings and to the corresponding bridge matrix. |
| Add a sub-process | Add its rows to the `All` sheet of `characterization_factors.xlsx`, and a column of the same name to the `All` sheet of `thresholds.xlsx` (budgets, units, conversions). |
| Change a threshold | `All` sheet of `thresholds.xlsx`. |
| Change the sharing principle | `sharing_principle` parameter of the plotting function (`None`, `"EPC"`, `"CTR"`); reference columns in `config.py`. |
| Regenerate the budget shares | Run `data/budget_shares.py`; keep the sheet names and the position of rows and columns, to which `thresholds.xlsx` is linked. |
| Change the trajectory of a domestic pressure | `my_manual_param.xlsx`, then `compute_ds_x_dom.py` and `run_pipeline_from_engine.ps1`. |
| Add a French scenario | Put the shock in `2-shocks/1-transitions_2050/`, add a column to `my_manual_param.xlsx` and an entry to `SCENARIO_FOLDER_NAMES` in `compute_ds_x_dom.py`. |
| Add a rest-of-world scenario | Put the IMAGE CSV file in `data/SSP/Scenarios/`, then run `run_all_SSPs.ps1`. For nitrogen and phosphorus, `Nutrients.xlsx` must contain the `PB_<name>` scenario. |
| Change a projection rule | `Matching` sheet of the matching workbook; constants of `generate_shock_dS.py` if an `A` group is renamed. |
| Change the base year of the SSP shocks | `path_in.accounts` of the template `planeFR/parameters/4-engine/settings_engine_SSPs.json`, then `run_all_SSPs.ps1`. |
| Change a calculation parameter of a MatMat step | The matching template of `planeFR/parameters/` (e.g. `m_row_mode` in the `gmrio_to_snac_s` templates). |
| Change EXIOBASE version | `-Version` parameter of the scripts, after adding the version to `SUPPORTED_VERSIONS_TO_DOI` if needed. |
| Create a scenario group to display | Declare it in `Module_PlaneFR/data/scenario_groups.json`, then run `build_scenario_groups.py`. |
| Run on another machine or folder layout | Keep `Module_PlaneFR` next to `MatMat`, or set `PLANEFR_MODULE_DIR`; adapt the two `config.toml` files. |
| Feed a group with household data | Add an entry `@{ Scenario = ...; Group = ... }` to `$Scenarios` in `BdF.ps1`. |

> **Possible improvement — tests.** PlanéFR has no automated tests, whereas MatMat already provides a `pytest` setup (`matmat-ademe/tests/`). The most useful targets, before any change to the module:
> - the formula parser of `extract_S_SSPs.py` (operators, units in parentheses, `OR`, `max`/`min`, yields);
> - `compute_ds` in `compute_ds_x_dom.py` and `generate_shock_dS.py`, on small synthetic tables;
> - `processing.filter_and_weight` and the footprint functions;
> - a non-regression test on the totals of the figures for a reference scenario. `export_impacts_coicop.py` (§ 8) already compares its totals with those of the figures, which can serve as a model.

---

## 12. Glossary

### MatMat variables

| Variable | Meaning |
|---|---|
| `x` | Output by product |
| `Z`, `A` | Intermediate consumption; technical coefficients |
| `Y` | Final demand |
| `L` | Leontief inverse |
| `K`, `Y_k`, `L_k` | Capital matrix, final demand and Leontief inverse when capital is endogenized |
| `F_x_dom` | Direct pressures of production, by sector |
| `S_x_dom` | Pressure intensity: `F_x_dom / x` |
| `F_Y`, `F_Z`, `S_Y`, `S_Z` | Pressures and intensities attached to final demand and to intermediate consumption, for the extensions of the France calibration |
| `F_Y_tot` | Direct pressures of final demand, summed by region (added by PlanéFR) |
| `M` | Multiplier: total pressure per unit of final demand |
| `M_RoW` | Pressure content of one imported unit |
| `d_imp` | Pressures embodied in imports |
| `d_cba` | Consumption footprint, by product and final demand category |
| suffix `_k` | Same quantity, computed with endogenized capital |
| prefix `d` (`dA`, `dY`, `dY_k`, `dS_x_dom`, `dS_Z`) | Relative shock: `W = W0 × (1 + dW)` |

### Terms

| Term | Meaning |
|---|---|
| Adapter | MatMat component converting an external database into MatMat accounts. |
| Pipeline | MatMat component transforming accounts into other accounts. |
| Engine | MatMat component applying a shock to accounts and computing them. |
| Extension | Environmental satellite account attached to accounts. |
| Sub-process | Planetary-boundary indicator displayed, obtained by characterizing one or several extensions. |
| SNAC | Single-country National Accounts Consistent: single-country accounts, imports separated from domestic production. |
| Scenario group | Sub-folder of `Module_PlaneFR/data/<version>/` gathering the scenarios displayed together. |
| CBA, PBA | Consumption-based accounting; production-based accounting. |
| EPC, CTR | Principles for sharing the global budget: equal per capita; capability to reduce. |
| DLS | Decent living standards: social floor. |
| IMAGE, IMAGE-GNM | Integrated assessment model providing the rest-of-world scenarios, and its nutrient module. |
| SSP | Shared Socioeconomic Pathways. |
| COICOP | Classification of household consumption by purpose. |
| HANPP | Human appropriation of net primary production. |
| PDF | Potentially disappeared fraction of species. |
| RMC | Raw material consumption. |
