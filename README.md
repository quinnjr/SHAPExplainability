# SHAPExplainability

A PluMA plugin for SHAP (SHapley Additive exPlanations) based feature attribution and machine learning model interpretability in multi-omics analysis.

## Overview

This plugin provides:

1. SHAP value computation for any sklearn-compatible classifier
2. Feature importance rankings across modalities
3. Cross-modal interaction analysis
4. Visualization outputs (summary plots, bar plots, dependence plots, modality comparisons)

## Installation

```bash
# Clone the repository
git clone https://github.com/quinnjr/SHAPExplainability.git

# Install dependencies
pip install -r requirements.txt
```

## Dependencies

- numpy
- pandas
- shap
- scikit-learn
- matplotlib
- scipy
- joblib

## Usage

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model` | Path to pickled sklearn model/pipeline | Required |
| `features` | Path to feature matrix (samples x features) | Required |
| `labels` | Path to sample labels CSV | Required |
| `explainer` | SHAP explainer: `tree`, `kernel`, `linear`, `auto` | `auto` |
| `background_samples` | Number of background samples for kernel SHAP | `100` |
| `n_top_features` | Number of top features to report | `20` |
| `compute_interactions` | Whether to compute SHAP interactions | `false` |

### Example Parameter File

```
model               models/trained_classifier.pkl
features            data/fused_features.csv
labels              data/sample_labels.csv
explainer           auto
background_samples  100
n_top_features      20
compute_interactions false
```

### Outputs

- SHAP values matrix (samples x features)
- Feature importance rankings
- Modality-wise importance breakdown
- Visualization plots (PNG: summary, bar, dependence, modality comparison)

## Testing

### Unit and integration tests (pytest)

```bash
pip install -r requirements-test.txt
pytest
```

Skip the slow integration tests with `pytest -m "not slow"`.

### PluMA contract verification

The `example/` directory ships a deterministic fixture based on the
UCI Parkinson's Voice Dataset (Little et al. 2007; 195 recordings x
22 biomedical voice features; 147 PD / 48 healthy) with modality
prefixes for vocal characteristics (`VC_`, 16 features) and
nonlinear dynamics (`ND_`, 6 features), and golden `.expected`
output files.

```bash
# Regenerate the fixture (downloads from archive.ics.uci.edu)
python scripts/fetch_test_data.py

# Run the plugin and compare against expected outputs
python scripts/verify_pluma.py
```

`verify_pluma.py` exits 0 when every generated file matches its
`.expected` twin within floating-point tolerance (EPS=1e-8, matching
PluMA's own `testPluMA.py` logic).

## Methods

### SHAP Values

SHAP values provide a unified measure of feature importance based on cooperative game theory. They decompose a prediction into additive contributions from each feature:

```
f(x) = φ₀ + Σᵢ φᵢ
```

Where φᵢ is the SHAP value for feature i, representing its contribution to the prediction.

### Explainer Types

- **TreeSHAP**: Exact, fast computation for tree-based models (Random Forest, XGBoost, LightGBM)
- **KernelSHAP**: Model-agnostic, uses weighted linear regression
- **LinearSHAP**: Exact computation for linear models

### Multi-omics Insights

SHAP values enable:
- Identifying which modality (metagenomics vs transcriptomics) drives predictions
- Understanding feature interactions across modalities
- Generating biologically interpretable explanations

## References

### SHAP Theory & Implementation

1. **Lundberg SM, Lee SI** (2017). A Unified Approach to Interpreting Model Predictions. *NIPS 2017*. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
   - *Core SHAP methodology - essential reference*

2. **Lundberg SM, Erion G, Chen H, et al.** (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2:56-67. [doi:10.1038/s42256-019-0138-9](https://doi.org/10.1038/s42256-019-0138-9)
   - *TreeSHAP algorithm and global explanations*

3. **Lundberg SM, Nair B, Vavilala MS, et al.** (2018). Explainable machine-learning predictions for the prevention of hypoxaemia during surgery. *Nature Biomedical Engineering*, 2:749-760. [doi:10.1038/s41551-018-0304-0](https://doi.org/10.1038/s41551-018-0304-0)
   - *Clinical application of SHAP*

### Shapley Values Foundation

4. **Shapley LS** (1953). A Value for N-Person Games. In: Kuhn HW, Tucker AW (eds) *Contributions to the Theory of Games II*. Princeton University Press. pp. 307-317. [doi:10.1515/9781400881970-018](https://doi.org/10.1515/9781400881970-018)
   - *Original Shapley value theory from game theory*

5. **Štrumbelj E, Kononenko I** (2014). Explaining prediction models and individual predictions with feature contributions. *Knowledge and Information Systems*, 41:647-665. [doi:10.1007/s10115-013-0679-x](https://doi.org/10.1007/s10115-013-0679-x)
   - *Shapley values applied to machine learning*

### Explainable AI

6. **Ribeiro MT, Singh S, Guestrin C** (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD '16*. [doi:10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)
   - *LIME methodology, conceptually related to SHAP*

7. **Murdoch WJ, Singh C, Kumbier K, Abbasi-Asl R, Yu B** (2019). Definitions, methods, and applications in interpretable machine learning. *PNAS*, 116(44):22071-22080. [doi:10.1073/pnas.1900654116](https://doi.org/10.1073/pnas.1900654116)
   - *Comprehensive review of interpretable ML*

### Test Fixture Dataset

8. **Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM** (2007). Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection. *Biomedical Engineering Online*, 6:23. [doi:10.1186/1475-925X-6-23](https://doi.org/10.1186/1475-925X-6-23)
   - *UCI Parkinsons Voice Dataset used by the committed `example/` fixture*

## Visualization Examples

The plugin generates several visualization types:

- **Summary Plot**: Global feature importance with distribution of SHAP values
- **Bar Plot**: Mean absolute SHAP values per feature
- **Dependence Plot**: SHAP value vs feature value for specific features
- **Modality Comparison**: Relative contribution of each omics modality

## License

MIT License

## Author

Joseph R. Quinn <quinn.josephr@protonmail.com>
