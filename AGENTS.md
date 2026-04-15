# AGENTS.md — SHAPExplainability

## Project Overview

PluMA plugin for SHAP (SHapley Additive exPlanations) based feature attribution on sklearn-compatible classifiers. Computes SHAP values, ranks feature importance globally and per-modality (metagenomics, transcriptomics, proteomics, metabolomics), optionally computes tree interaction values, and produces CSV/numpy outputs plus PNG visualizations. Designed for multi-omics ML interpretability.

## Architecture

Single-module plugin following the PluMA plugin contract:

- `SHAPExplainability.py` — the core class `SHAPExplainability` with `input()` / `run()` / `output()` lifecycle and private `_snake_case` helpers for explainer creation, SHAP computation, importance aggregation, modality extraction, and visualization.
- `SHAPExplainabilityPlugin.py` — thin subclass named per PluMA convention (`<Name>Plugin.py` with class `<Name>Plugin`) so the PluMA Python loader can instantiate it. Delegates all behavior to the core class.
- `test_shap_explainability.py` — pytest suite with synthetic data and sklearn models.
- `scripts/release.py` — semver tagging and optional GitHub release via `gh`.
- `scripts/fetch_test_data.py` — downloads and prepares the example fixture.
- `scripts/verify_pluma.py` — runs the plugin as PluMA would and compares outputs against `example/*.expected`.
- `parameters.shap.txt` — example parameter file.

No package structure, no `__init__.py`, no CLI entry point. The class is instantiated by PluMA's plugin loader.

## Conventions

- **Language:** Python 3. Uses `from __future__ import annotations`, `Literal`, and `X | Y` union syntax.
- **Naming:** Class name matches the filename exactly (`SHAPExplainability`). Private methods use `_snake_case`.
- **Types:** `ExplainerType = Literal["tree", "kernel", "linear", "deep", "auto"]`. Instance attributes are typed in `__init__`.
- **Docstrings:** Google-style with `Args:` / `Returns:` sections on all methods.
- **Dependencies:** Pinned minimum versions in `requirements.txt`. No build system or packaging manifest.
- **Plotting:** Uses matplotlib with the `Agg` (non-interactive) backend.
- **Model loading:** The plugin deserializes sklearn models/pipelines from files via the `input()` method. Models are expected to be sklearn-compatible with `predict_proba` or `decision_function`.
- **No MuPDF.** Use Micropdf if PDF handling is ever needed.

## Testing

- **Framework:** pytest (config in `pytest.ini`).
- **Run tests:** `pip install -r requirements-test.txt && pytest`
- **Test deps:** `pytest>=7.4.0`, `pytest-cov>=4.1.0` (via `requirements-test.txt`).
- **Markers:** `slow` for expensive tests (`-m "not slow"` to skip).
- **Pattern:** Fixtures create synthetic feature matrices and train LogisticRegression/RandomForest models; test classes are `Test*`, functions are `test_*`.

## Parameter File Format

Whitespace-delimited key-value file (one pair per line, `#` comments ignored). Keys: `model` (path to serialized sklearn model/pipeline), `features` (CSV, samples x features, `index_col=0`), `labels` (CSV, first column), `explainer` (`tree`/`kernel`/`linear`/`deep`/`auto`), `background_samples`, `n_top_features`, `compute_interactions` (`true`/`false`).

## Pipeline Steps (run method)

1. **Create explainer** — auto-detect model type or use explicit `explainer` param. Unwraps sklearn `Pipeline` to get the final estimator. Tree -> `TreeExplainer`, linear -> `LinearExplainer`, fallback -> `KernelExplainer` (uses background sample subset).
2. **Compute SHAP values** — calls `explainer.shap_values(X)`. For binary classification, takes positive-class values.
3. **Feature importance** — mean |SHAP| per feature, sorted descending, with modality label.
4. **Modality importance** — groups features by prefix (`MG_` -> metagenomics, `TX_` -> transcriptomics, `PT_` -> proteomics, `MT_` -> metabolomics; unknown prefix kept as-is; no `_` -> `"unknown"`). Aggregates sum, mean, std, count, relative contribution %, top-5 features per modality.
5. **Interaction values** (optional, tree only) — `shap_interaction_values`.

## Output Files

All outputs derive from the base path passed to `output()` using `Path.with_suffix()`:

- `*.shap_values.csv` — SHAP values matrix (samples x features)
- `*.feature_importance.csv` — ranked feature importance with modality labels
- `*.modality_importance.csv` — per-modality aggregated importance
- `*.interaction_values.npy` — interaction values (only if `compute_interactions=true` and explainer is tree)
- `*.summary_plot.png` — beeswarm summary plot
- `*.bar_plot.png` — mean |SHAP| bar plot
- `*.dependence_<i>_<feature>.png` — dependence plots for top 5 features
- `*.modality_comparison.png` — bar chart of relative modality contributions
- `*.summary.txt` — human-readable text summary

## Known TODOs and Discrepancies

- **`deep` explainer:** Listed in `ExplainerType` and documented in README, but raises `NotImplementedError` at runtime.
- **Force plots / HTML:** README lists force plots and HTML output; the code only generates PNG (summary, bar, dependence, modality comparison). No `shap.force_plot` or HTML export exists.
- **`scipy`:** Used at runtime (`scipy.special.expit` in `_predict_proba` fallback) but not listed in README's dependency bullets (it is in `requirements.txt`).
- **No `LICENSE` file** in the directory despite README stating MIT.

## Release

```
python scripts/release.py patch|minor|major
```

Expects a clean git working tree. Tags with `vX.Y.Z`, optionally pushes and creates a GitHub release.

## Attribution

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT
