"""
Unit tests for SHAPExplainability PluMA Plugin.

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
"""

import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from SHAPExplainability import SHAPExplainability


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """Create sample feature matrix with modality prefixes."""
    np.random.seed(42)
    n_samples = 50
    
    # Metagenomics features
    mg_features = np.random.rand(n_samples, 10)
    mg_cols = [f"MG_Taxa_{i}" for i in range(10)]
    
    # Transcriptomics features
    tx_features = np.random.rand(n_samples, 15)
    tx_cols = [f"TX_Gene_{i}" for i in range(15)]
    
    data = np.hstack([mg_features, tx_features])
    columns = mg_cols + tx_cols
    sample_names = [f"Sample_{i:03d}" for i in range(n_samples)]
    
    return pd.DataFrame(data, index=sample_names, columns=columns)


@pytest.fixture
def sample_labels() -> pd.DataFrame:
    """Create sample labels."""
    np.random.seed(42)
    sample_names = [f"Sample_{i:03d}" for i in range(50)]
    labels = [0] * 25 + [1] * 25
    
    return pd.DataFrame({"label": labels}, index=sample_names)


@pytest.fixture
def trained_logistic_model(sample_features, sample_labels):
    """Create a trained logistic regression model."""
    X = sample_features.values
    y = sample_labels["label"].values
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])
    pipeline.fit(X, y)
    
    return pipeline


@pytest.fixture
def trained_rf_model(sample_features, sample_labels):
    """Create a trained random forest model."""
    X = sample_features.values
    y = sample_labels["label"].values
    
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def plugin_with_data(temp_dir, sample_features, sample_labels, trained_logistic_model):
    """Create plugin instance with loaded test data."""
    features_path = temp_dir / "features.csv"
    labels_path = temp_dir / "labels.csv"
    model_path = temp_dir / "model.pkl"
    
    sample_features.to_csv(features_path)
    sample_labels.to_csv(labels_path)
    
    with open(model_path, "wb") as f:
        pickle.dump(trained_logistic_model, f)
    
    param_path = temp_dir / "params.txt"
    param_path.write_text(f"""model\t{model_path}
features\t{features_path}
labels\t{labels_path}
explainer\tauto
background_samples\t20
n_top_features\t10
compute_interactions\tfalse
""")
    
    plugin = SHAPExplainability()
    plugin.input(str(param_path))
    
    return plugin


class TestSHAPInit:
    """Tests for plugin initialization."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        plugin = SHAPExplainability()
        
        assert plugin.explainer_type == "auto"
        assert plugin.background_samples == 100
        assert plugin.n_top_features == 20
        assert plugin.compute_interactions is False
    
    def test_load_model(self, temp_dir, trained_logistic_model):
        """Test loading pickled model."""
        model_path = temp_dir / "model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_logistic_model, f)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"model\t{model_path}\n")
        
        plugin = SHAPExplainability()
        plugin.input(str(param_path))
        
        assert plugin.model is not None
    
    def test_load_features(self, temp_dir, sample_features):
        """Test loading feature matrix."""
        features_path = temp_dir / "features.csv"
        sample_features.to_csv(features_path)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"features\t{features_path}\n")
        
        plugin = SHAPExplainability()
        plugin.input(str(param_path))
        
        assert plugin.features is not None
        assert plugin.features.shape == sample_features.shape


class TestExplainerDetection:
    """Tests for explainer type detection."""
    
    def test_detect_linear_explainer(self, plugin_with_data):
        """Test detection of linear explainer for logistic regression."""
        plugin = plugin_with_data
        
        explainer_type = plugin._detect_explainer_type()
        
        assert explainer_type == "linear"
    
    def test_detect_tree_explainer(self, temp_dir, sample_features, sample_labels, trained_rf_model):
        """Test detection of tree explainer for random forest."""
        model_path = temp_dir / "model.pkl"
        features_path = temp_dir / "features.csv"
        
        with open(model_path, "wb") as f:
            pickle.dump(trained_rf_model, f)
        sample_features.to_csv(features_path)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"""model\t{model_path}
features\t{features_path}
explainer\tauto
""")
        
        plugin = SHAPExplainability()
        plugin.input(str(param_path))
        
        explainer_type = plugin._detect_explainer_type()
        
        assert explainer_type == "tree"


class TestUnderlyingModel:
    """Tests for extracting underlying model from pipeline."""
    
    def test_extract_from_pipeline(self, plugin_with_data):
        """Test extracting classifier from sklearn pipeline."""
        plugin = plugin_with_data
        
        model = plugin._get_underlying_model()
        
        assert isinstance(model, LogisticRegression)
    
    def test_return_model_directly(self, temp_dir, sample_features, trained_rf_model):
        """Test returning model directly if not a pipeline."""
        model_path = temp_dir / "model.pkl"
        features_path = temp_dir / "features.csv"
        
        with open(model_path, "wb") as f:
            pickle.dump(trained_rf_model, f)
        sample_features.to_csv(features_path)
        
        param_path = temp_dir / "params.txt"
        param_path.write_text(f"""model\t{model_path}
features\t{features_path}
""")
        
        plugin = SHAPExplainability()
        plugin.input(str(param_path))
        
        model = plugin._get_underlying_model()
        
        assert isinstance(model, RandomForestClassifier)


class TestModalityExtraction:
    """Tests for modality extraction from feature names."""
    
    def test_extract_metagenomics_modality(self):
        """Test extracting metagenomics modality."""
        plugin = SHAPExplainability()
        
        modality = plugin._extract_modality("MG_Taxa_001")
        
        assert modality == "metagenomics"
    
    def test_extract_transcriptomics_modality(self):
        """Test extracting transcriptomics modality."""
        plugin = SHAPExplainability()
        
        modality = plugin._extract_modality("TX_Gene_001")
        
        assert modality == "transcriptomics"
    
    def test_extract_unknown_modality(self):
        """Test extracting unknown modality prefix."""
        plugin = SHAPExplainability()
        
        modality = plugin._extract_modality("OTHER_Feature")
        
        assert modality == "OTHER"
    
    def test_no_prefix(self):
        """Test feature with no modality prefix."""
        plugin = SHAPExplainability()
        
        modality = plugin._extract_modality("FeatureWithNoPrefix")
        
        assert modality == "unknown"


class TestSHAPValues:
    """Tests for SHAP value computation."""
    
    @pytest.mark.slow
    def test_shap_values_shape(self, plugin_with_data):
        """Test SHAP values have correct shape."""
        plugin = plugin_with_data
        plugin.explainer = plugin._create_explainer()
        
        shap_values, expected_value = plugin._compute_shap_values()
        
        assert shap_values.shape == plugin.features.shape
    
    @pytest.mark.slow
    def test_expected_value_scalar(self, plugin_with_data):
        """Test expected value is a scalar or array."""
        plugin = plugin_with_data
        plugin.explainer = plugin._create_explainer()
        
        _, expected_value = plugin._compute_shap_values()
        
        assert expected_value is not None


class TestFeatureImportance:
    """Tests for feature importance computation."""
    
    @pytest.mark.slow
    def test_feature_importance_structure(self, plugin_with_data):
        """Test feature importance DataFrame structure."""
        plugin = plugin_with_data
        plugin.run()
        
        assert "feature" in plugin.feature_importance.columns
        assert "mean_abs_shap" in plugin.feature_importance.columns
        assert "modality" in plugin.feature_importance.columns
        assert "rank" in plugin.feature_importance.columns
    
    @pytest.mark.slow
    def test_feature_importance_sorted(self, plugin_with_data):
        """Test feature importance is sorted by importance."""
        plugin = plugin_with_data
        plugin.run()
        
        importances = plugin.feature_importance["mean_abs_shap"].values
        
        # Check descending order
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))


class TestModalityImportance:
    """Tests for modality importance aggregation."""
    
    @pytest.mark.slow
    def test_modality_importance_structure(self, plugin_with_data):
        """Test modality importance DataFrame structure."""
        plugin = plugin_with_data
        plugin.run()
        
        assert "modality" in plugin.modality_importance.columns
        assert "total_importance" in plugin.modality_importance.columns
        assert "relative_contribution" in plugin.modality_importance.columns
    
    @pytest.mark.slow
    def test_relative_contribution_sums_to_100(self, plugin_with_data):
        """Test relative contributions sum to 100%."""
        plugin = plugin_with_data
        plugin.run()
        
        total = plugin.modality_importance["relative_contribution"].sum()
        
        np.testing.assert_almost_equal(total, 100.0, decimal=1)


class TestRunPipeline:
    """Tests for full pipeline execution."""
    
    @pytest.mark.slow
    def test_run_completes(self, plugin_with_data):
        """Test that run() completes without error."""
        plugin = plugin_with_data
        
        plugin.run()
        
        assert plugin.shap_values is not None
        assert plugin.feature_importance is not None
        assert plugin.modality_importance is not None


class TestOutput:
    """Tests for output generation."""
    
    @pytest.mark.slow
    def test_output_creates_files(self, plugin_with_data, temp_dir):
        """Test that output creates expected files."""
        plugin = plugin_with_data
        plugin.run()
        
        output_path = temp_dir / "output"
        plugin.output(str(output_path))
        
        assert output_path.with_suffix(".shap_values.csv").exists()
        assert output_path.with_suffix(".feature_importance.csv").exists()
        assert output_path.with_suffix(".modality_importance.csv").exists()
        assert output_path.with_suffix(".summary.txt").exists()
    
    @pytest.mark.slow
    def test_output_creates_plots(self, plugin_with_data, temp_dir):
        """Test that visualization plots are created."""
        plugin = plugin_with_data
        plugin.run()
        
        output_path = temp_dir / "output"
        plugin.output(str(output_path))
        
        assert output_path.with_suffix(".summary_plot.png").exists()
        assert output_path.with_suffix(".bar_plot.png").exists()


class TestParameterParsing:
    """Tests for parameter file parsing tolerance."""

    def test_tab_separated_parameters(self, temp_dir):
        param_path = temp_dir / "params.txt"
        param_path.write_text("explainer\ttree\nbackground_samples\t50\n")

        plugin = SHAPExplainability()
        plugin.input(str(param_path))

        assert plugin.explainer_type == "tree"
        assert plugin.background_samples == 50

    def test_space_separated_parameters(self, temp_dir):
        param_path = temp_dir / "params.txt"
        param_path.write_text("explainer    tree\nbackground_samples   50\n")

        plugin = SHAPExplainability()
        plugin.input(str(param_path))

        assert plugin.explainer_type == "tree"
        assert plugin.background_samples == 50

    def test_comment_lines_ignored(self, temp_dir):
        param_path = temp_dir / "params.txt"
        param_path.write_text("# a comment\nexplainer\ttree\n# another\n")

        plugin = SHAPExplainability()
        plugin.input(str(param_path))

        assert plugin.explainer_type == "tree"


class TestWriteSummary:
    """Tests for summary text file generation."""

    def test_expected_value_array_does_not_crash(self, temp_dir):
        """_write_summary must handle array-valued expected_value."""
        plugin = SHAPExplainability()
        plugin.features = pd.DataFrame(
            np.zeros((3, 2)),
            columns=["MG_a", "TX_b"],
            index=["s0", "s1", "s2"],
        )
        plugin.expected_value = np.array([0.3, 0.7])
        plugin.explainer_type = "tree"

        out = temp_dir / "summary.txt"
        plugin._write_summary(out)
        assert out.exists()
        text = out.read_text()
        assert "Expected value" in text


class TestNormalizeShapOutput:
    """Tests for SHAP output normalization across versions."""

    def test_list_binary_returns_positive_class(self):
        plugin = SHAPExplainability()
        raw = [np.zeros((5, 3)), np.ones((5, 3))]
        normalized = plugin._normalize_shap_output(raw)
        assert normalized.shape == (5, 3)
        assert np.all(normalized == 1.0)

    def test_list_multiclass_returns_positive_class(self):
        plugin = SHAPExplainability()
        raw = [np.zeros((5, 3)), np.ones((5, 3)), np.full((5, 3), 2.0)]
        normalized = plugin._normalize_shap_output(raw)
        assert normalized.shape == (5, 3)
        assert np.all(normalized == 1.0)

    def test_2d_array_passes_through(self):
        plugin = SHAPExplainability()
        raw = np.arange(15).reshape(5, 3).astype(float)
        normalized = plugin._normalize_shap_output(raw)
        assert normalized.shape == (5, 3)
        assert np.array_equal(normalized, raw)

    def test_3d_binary_returns_positive_slice(self):
        plugin = SHAPExplainability()
        raw = np.zeros((5, 3, 2))
        raw[..., 1] = 1.0
        normalized = plugin._normalize_shap_output(raw)
        assert normalized.shape == (5, 3)
        assert np.all(normalized == 1.0)

    def test_3d_multiclass_returns_positive_slice(self):
        plugin = SHAPExplainability()
        raw = np.zeros((5, 3, 4))
        raw[..., 1] = 1.0
        normalized = plugin._normalize_shap_output(raw)
        assert normalized.shape == (5, 3)
        assert np.all(normalized == 1.0)

    def test_explanation_object(self):
        class FakeExplanation:
            values = np.ones((5, 3))
        plugin = SHAPExplainability()
        normalized = plugin._normalize_shap_output(FakeExplanation())
        assert normalized.shape == (5, 3)
        assert np.all(normalized == 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
