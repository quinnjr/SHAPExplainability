"""
SHAPExplainability PluMA Plugin

SHAP (SHapley Additive exPlanations) based feature attribution for 
machine learning model interpretability in multi-omics analysis.

This plugin provides:
1. SHAP value computation for any sklearn-compatible classifier
2. Feature importance rankings across modalities
3. Cross-modal interaction analysis
4. Visualization outputs (summary plots, force plots, dependence plots)

Author: Joseph R. Quinn <quinn.josephr@protonmail.com>
License: MIT

References:
    SHAP Theory & Implementation:
    - Lundberg SM, Lee SI (2017). A Unified Approach to Interpreting Model 
      Predictions. NIPS 2017. arXiv:1705.07874 [Core SHAP methodology]
    
    - Lundberg SM, Erion G, Chen H, et al. (2020). From local explanations to 
      global understanding with explainable AI for trees. Nat Mach Intell. 
      2:56-67. doi:10.1038/s42256-019-0138-9 [TreeSHAP]
    
    - Lundberg SM, Nair B, Vavilala MS, et al. (2018). Explainable machine-
      learning predictions for the prevention of hypoxaemia during surgery. 
      Nat Biomed Eng. 2:749-760. doi:10.1038/s41551-018-0304-0 [Clinical SHAP]
    
    Shapley Values Foundation:
    - Shapley LS (1953). A Value for N-Person Games. In: Kuhn HW, Tucker AW 
      (eds) Contributions to the Theory of Games II. Princeton University 
      Press. pp. 307-317. doi:10.1515/9781400881970-018 [Original Shapley]
    
    - Strumbelj E, Kononenko I (2014). Explaining prediction models and 
      individual predictions with feature contributions. Knowl Inf Syst. 
      41:647-665. doi:10.1007/s10115-013-0679-x [Shapley for ML]
    
    Explainable AI in Bioinformatics:
    - Ribeiro MT, Singh S, Guestrin C (2016). "Why Should I Trust You?": 
      Explaining the Predictions of Any Classifier. KDD '16. 
      doi:10.1145/2939672.2939778 [LIME, related to SHAP]
    
    - Murdoch WJ, Singh C, Kumbier K, Abbasi-Asl R, Yu B (2019). Definitions, 
      methods, and applications in interpretable machine learning. PNAS. 
      116(44):22071-22080. doi:10.1073/pnas.1900654116 [XAI review]
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd


ExplainerType = Literal["tree", "kernel", "linear", "deep", "auto"]


class SHAPExplainability:
    """
    PluMA plugin for SHAP-based model explainability.
    
    Computes SHAP values for trained classifiers to identify which
    features (microbial taxa, genes) drive predictions in PD classification.
    
    Parameters (via input file):
        model: Path to pickled sklearn model/pipeline
        features: Path to feature matrix (samples x features)
        labels: Path to sample labels CSV
        explainer: SHAP explainer type ("tree", "kernel", "linear", "deep", "auto")
        background_samples: Number of background samples for kernel SHAP (default: 100)
        n_top_features: Number of top features to report (default: 20)
        compute_interactions: Whether to compute SHAP interactions (default: false)
    
    Outputs:
        - SHAP values matrix (samples x features)
        - Feature importance rankings
        - Per-modality importance summary
        - Visualization plots (PNG/HTML)
    """
    
    def __init__(self) -> None:
        """Initialize plugin state."""
        self.parameters: dict[str, str] = {}
        
        # Data
        self.model: Any = None
        self.features: pd.DataFrame | None = None
        self.labels: pd.Series | None = None
        
        # Results
        self.shap_values: np.ndarray | None = None
        self.expected_value: float | np.ndarray | None = None
        self.feature_importance: pd.DataFrame | None = None
        self.modality_importance: pd.DataFrame | None = None
        self.interaction_values: np.ndarray | None = None
        self.explainer: Any = None
        
        # Default parameters
        self.explainer_type: ExplainerType = "auto"
        self.background_samples: int = 100
        self.n_top_features: int = 20
        self.compute_interactions: bool = False
    
    def input(self, filename: str) -> None:
        """
        Load input data and configuration parameters.
        
        Args:
            filename: Path to parameter file with key-value pairs
        """
        param_path = Path(filename)
        
        # Parse parameter file
        with param_path.open() as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        self.parameters[parts[0]] = parts[1]
        
        # Load trained model
        if "model" in self.parameters:
            with open(self.parameters["model"], "rb") as f:
                self.model = pickle.load(f)
        
        # Load feature matrix
        if "features" in self.parameters:
            self.features = pd.read_csv(
                self.parameters["features"], 
                index_col=0
            )
        
        # Load sample labels
        if "labels" in self.parameters:
            labels_df = pd.read_csv(self.parameters["labels"], index_col=0)
            self.labels = labels_df.iloc[:, 0]
        
        # Parse optional parameters
        if "explainer" in self.parameters:
            self.explainer_type = self.parameters["explainer"].lower()  # type: ignore
        
        if "background_samples" in self.parameters:
            self.background_samples = int(self.parameters["background_samples"])
        
        if "n_top_features" in self.parameters:
            self.n_top_features = int(self.parameters["n_top_features"])
        
        if "compute_interactions" in self.parameters:
            self.compute_interactions = self.parameters["compute_interactions"].lower() == "true"
    
    def run(self) -> None:
        """
        Execute SHAP explainability analysis.
        
        Steps:
        1. Initialize appropriate SHAP explainer
        2. Compute SHAP values for all samples
        3. Calculate feature importance rankings
        4. Aggregate importance by modality
        5. Compute interaction values (optional)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Check 'model' parameter.")
        
        if self.features is None:
            raise ValueError("Features not loaded. Check 'features' parameter.")
        
        # Step 1: Initialize explainer
        self.explainer = self._create_explainer()
        
        # Step 2: Compute SHAP values
        self.shap_values, self.expected_value = self._compute_shap_values()
        
        # Step 3: Calculate feature importance
        self.feature_importance = self._compute_feature_importance()
        
        # Step 4: Aggregate by modality
        self.modality_importance = self._compute_modality_importance()
        
        # Step 5: Interaction values (optional)
        if self.compute_interactions:
            self.interaction_values = self._compute_interactions()
    
    def output(self, filename: str) -> None:
        """
        Write results to output files.
        
        Args:
            filename: Base path for output files
        """
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write SHAP values matrix
        if self.shap_values is not None:
            shap_df = pd.DataFrame(
                self.shap_values,
                index=self.features.index,  # type: ignore
                columns=self.features.columns  # type: ignore
            )
            shap_df.to_csv(output_path.with_suffix(".shap_values.csv"))
        
        # Write feature importance
        if self.feature_importance is not None:
            self.feature_importance.to_csv(
                output_path.with_suffix(".feature_importance.csv")
            )
        
        # Write modality importance
        if self.modality_importance is not None:
            self.modality_importance.to_csv(
                output_path.with_suffix(".modality_importance.csv")
            )
        
        # Write interaction values (if computed)
        if self.interaction_values is not None:
            # Save as numpy for efficiency
            np.save(
                output_path.with_suffix(".interaction_values.npy"),
                self.interaction_values
            )
        
        # Generate visualizations
        self._generate_visualizations(output_path)
        
        # Write summary
        self._write_summary(output_path.with_suffix(".summary.txt"))
    
    def _create_explainer(self) -> Any:
        """
        Create appropriate SHAP explainer based on model type.
        
        Returns:
            SHAP explainer instance
        """
        import shap
        
        X = self.features.values  # type: ignore
        
        # Auto-detect explainer type if not specified
        if self.explainer_type == "auto":
            self.explainer_type = self._detect_explainer_type()
        
        if self.explainer_type == "tree":
            # For tree-based models (RF, XGBoost, LightGBM)
            model = self._get_underlying_model()
            return shap.TreeExplainer(model)
        
        elif self.explainer_type == "linear":
            # For linear models (logistic regression, linear SVM)
            model = self._get_underlying_model()
            return shap.LinearExplainer(model, X)
        
        elif self.explainer_type == "kernel":
            # For any model (model-agnostic)
            # Use background sample for approximation
            background_idx = np.random.choice(
                X.shape[0], 
                min(self.background_samples, X.shape[0]),
                replace=False
            )
            background = X[background_idx]
            
            return shap.KernelExplainer(
                self._predict_proba,
                background
            )
        
        elif self.explainer_type == "deep":
            # For deep learning models
            # TODO: Implement deep explainer
            raise NotImplementedError("Deep explainer not yet implemented")
        
        else:
            raise ValueError(f"Unknown explainer type: {self.explainer_type}")
    
    def _detect_explainer_type(self) -> ExplainerType:
        """
        Auto-detect appropriate explainer based on model type.
        
        Returns:
            Recommended explainer type
        """
        model = self._get_underlying_model()
        model_class = type(model).__name__.lower()
        
        # Tree-based models
        tree_models = [
            "randomforest", "gradientboosting", "xgb", "lgb", 
            "catboost", "extratrees", "decisiontree"
        ]
        if any(t in model_class for t in tree_models):
            return "tree"
        
        # Linear models
        linear_models = ["logistic", "linear", "ridge", "lasso", "elasticnet"]
        if any(l in model_class for l in linear_models):
            return "linear"
        
        # Default to kernel (model-agnostic)
        return "kernel"
    
    def _get_underlying_model(self) -> Any:
        """
        Extract the underlying model from a pipeline if needed.
        
        Returns:
            The actual classifier/model
        """
        from sklearn.pipeline import Pipeline
        
        if isinstance(self.model, Pipeline):
            # Get last step (usually the classifier)
            return self.model.steps[-1][1]
        
        return self.model
    
    def _predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Prediction function for kernel SHAP.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probability predictions
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba, use decision function
            decisions = self.model.decision_function(X)
            # Convert to probabilities
            from scipy.special import expit
            probs = expit(decisions)
            return np.column_stack([1 - probs, probs])
    
    def _compute_shap_values(self) -> tuple[np.ndarray, float | np.ndarray]:
        """
        Compute SHAP values for all samples.
        
        Returns:
            Tuple of (SHAP values array, expected value)
        """
        X = self.features.values  # type: ignore
        
        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle multi-class case
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            shap_values = shap_values[1]
        
        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)) and len(expected_value) > 1:
            expected_value = expected_value[1]
        
        return shap_values, expected_value
    
    def _compute_feature_importance(self) -> pd.DataFrame:
        """
        Compute global feature importance from SHAP values.
        
        Returns:
            DataFrame with feature importance rankings
        """
        feature_names = self.features.columns.tolist()  # type: ignore
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)  # type: ignore
        
        # Standard deviation for error bars
        std_shap = np.abs(self.shap_values).std(axis=0)  # type: ignore
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
            "std_shap": std_shap,
            "modality": [self._extract_modality(f) for f in feature_names]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values(
            "mean_abs_shap", 
            ascending=False
        ).reset_index(drop=True)
        
        # Add rank
        importance_df["rank"] = range(1, len(importance_df) + 1)
        
        return importance_df
    
    def _extract_modality(self, feature_name: str) -> str:
        """
        Extract modality from feature name.
        
        Assumes features are prefixed with modality (e.g., MG_, TX_).
        
        Args:
            feature_name: Feature name with modality prefix
            
        Returns:
            Modality name
        """
        if "_" in feature_name:
            prefix = feature_name.split("_")[0]
            modality_map = {
                "MG": "metagenomics",
                "TX": "transcriptomics",
                "PT": "proteomics",
                "MT": "metabolomics"
            }
            return modality_map.get(prefix, prefix)
        
        return "unknown"
    
    def _compute_modality_importance(self) -> pd.DataFrame:
        """
        Aggregate importance by modality.
        
        Returns:
            DataFrame with per-modality importance statistics
        """
        if self.feature_importance is None:
            return pd.DataFrame()
        
        modality_stats = self.feature_importance.groupby("modality").agg({
            "mean_abs_shap": ["sum", "mean", "std", "count"],
            "feature": lambda x: list(x[:5])  # Top 5 features per modality
        }).reset_index()
        
        # Flatten column names
        modality_stats.columns = [
            "modality", "total_importance", "mean_importance", 
            "std_importance", "n_features", "top_features"
        ]
        
        # Calculate relative contribution
        total = modality_stats["total_importance"].sum()
        modality_stats["relative_contribution"] = (
            modality_stats["total_importance"] / total * 100
        )
        
        # Sort by total importance
        modality_stats = modality_stats.sort_values(
            "total_importance", 
            ascending=False
        )
        
        return modality_stats
    
    def _compute_interactions(self) -> np.ndarray:
        """
        Compute SHAP interaction values.
        
        Returns:
            Interaction values array (n_samples x n_features x n_features)
        """
        # Only available for tree explainer
        if self.explainer_type != "tree":
            print("Warning: Interaction values only available for tree explainer")
            return np.array([])
        
        X = self.features.values  # type: ignore
        interaction_values = self.explainer.shap_interaction_values(X)
        
        # Handle multi-class case
        if isinstance(interaction_values, list):
            interaction_values = interaction_values[1]
        
        return interaction_values
    
    def _generate_visualizations(self, output_path: Path) -> None:
        """
        Generate SHAP visualization plots.
        
        Args:
            output_path: Base path for output files
        """
        import shap
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
        
        X = self.features  # Keep as DataFrame for feature names
        
        # 1. Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, X,
            show=False,
            max_display=self.n_top_features
        )
        plt.tight_layout()
        plt.savefig(output_path.with_suffix(".summary_plot.png"), dpi=150)
        plt.close()
        
        # 2. Bar plot (mean absolute SHAP)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, X,
            plot_type="bar",
            show=False,
            max_display=self.n_top_features
        )
        plt.tight_layout()
        plt.savefig(output_path.with_suffix(".bar_plot.png"), dpi=150)
        plt.close()
        
        # 3. Top feature dependence plots
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(5)["feature"].tolist()
            
            for i, feature in enumerate(top_features):
                plt.figure(figsize=(8, 6))
                feature_idx = list(X.columns).index(feature)
                shap.dependence_plot(
                    feature_idx, self.shap_values, X,
                    show=False
                )
                plt.tight_layout()
                plt.savefig(
                    output_path.with_suffix(f".dependence_{i+1}_{feature[:20]}.png"),
                    dpi=150
                )
                plt.close()
        
        # 4. Modality comparison plot
        if self.modality_importance is not None:
            plt.figure(figsize=(10, 6))
            modalities = self.modality_importance["modality"].tolist()
            contributions = self.modality_importance["relative_contribution"].tolist()
            
            colors = plt.cm.Set2(np.linspace(0, 1, len(modalities)))  # type: ignore
            plt.bar(modalities, contributions, color=colors)
            plt.xlabel("Modality")
            plt.ylabel("Relative Contribution (%)")
            plt.title("Feature Importance by Modality")
            plt.tight_layout()
            plt.savefig(output_path.with_suffix(".modality_comparison.png"), dpi=150)
            plt.close()
    
    def _write_summary(self, filepath: Path) -> None:
        """
        Write explainability summary.
        
        Args:
            filepath: Path for summary file
        """
        with filepath.open("w") as f:
            f.write("SHAPExplainability Summary\n")
            f.write("=" * 40 + "\n\n")
            
            if self.features is not None:
                f.write(f"Number of samples: {self.features.shape[0]}\n")
                f.write(f"Number of features: {self.features.shape[1]}\n")
            
            f.write(f"Explainer type: {self.explainer_type}\n")
            
            if self.expected_value is not None:
                f.write(f"Expected value (base rate): {self.expected_value:.4f}\n")
            
            if self.feature_importance is not None:
                f.write(f"\nTop {self.n_top_features} Most Important Features:\n")
                f.write("-" * 40 + "\n")
                
                for _, row in self.feature_importance.head(self.n_top_features).iterrows():
                    f.write(
                        f"  {row['rank']:3d}. {row['feature'][:40]:40s} "
                        f"({row['modality']:15s}): {row['mean_abs_shap']:.4f}\n"
                    )
            
            if self.modality_importance is not None:
                f.write(f"\nImportance by Modality:\n")
                f.write("-" * 40 + "\n")
                
                for _, row in self.modality_importance.iterrows():
                    f.write(
                        f"  {row['modality']:15s}: {row['relative_contribution']:.1f}% "
                        f"({row['n_features']} features)\n"
                    )
            
            f.write(f"\nParameters:\n")
            f.write(f"  explainer: {self.explainer_type}\n")
            f.write(f"  background_samples: {self.background_samples}\n")
            f.write(f"  n_top_features: {self.n_top_features}\n")
            f.write(f"  compute_interactions: {self.compute_interactions}\n")
