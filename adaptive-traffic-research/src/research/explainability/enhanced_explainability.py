"""
Enhanced Explainable AI System.

Phase 5: Integration of SHAP, LIME, and counterfactual explanations
for world-class model interpretability.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Try to import explainability libraries with graceful fallbacks
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. SHAP explanations will be disabled.")

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    logger.warning("LIME not available. LIME explanations will be disabled.")


class EnhancedExplainer:
    """
    Enhanced explainability system with multiple explanation methods.
    
    Supports:
    - SHAP values (global and local)
    - LIME explanations
    - Counterfactual explanations
    - Feature importance
    - Natural language explanations
    """
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
    ):
        """
        Initialize enhanced explainer.
        
        Args:
            model: Model to explain
            feature_names: Names of input features
            background_data: Background dataset for SHAP
        """
        self.model = model
        self.feature_names = feature_names or [f"feature_{i}" for i in range(100)]
        self.background_data = background_data
        
        # Initialize SHAP explainer
        self.shap_explainer = None
        if SHAP_AVAILABLE and background_data is not None:
            try:
                # Try TreeExplainer for tree-based models
                if hasattr(model, 'predict_proba'):
                    self.shap_explainer = shap.TreeExplainer(model)
                # Try KernelExplainer as fallback
                elif hasattr(model, 'predict'):
                    self.shap_explainer = shap.KernelExplainer(
                        model.predict,
                        background_data,
                    )
                else:
                    logger.warning("Unknown model type for SHAP")
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer: {e}")
        
        # Initialize LIME explainer
        self.lime_explainer = None
        if LIME_AVAILABLE and background_data is not None:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    background_data,
                    feature_names=self.feature_names,
                    mode='regression',
                )
            except Exception as e:
                logger.warning(f"Failed to create LIME explainer: {e}")
    
    def explain_shap(
        self,
        instance: np.ndarray,
        global_explanation: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate SHAP explanations.
        
        Args:
            instance: Instance to explain
            global_explanation: Whether to generate global explanation
            
        Returns:
            Explanation dictionary with SHAP values and feature importance
        """
        if not SHAP_AVAILABLE or self.shap_explainer is None:
            logger.warning("SHAP explanations not available")
            return {"error": "SHAP not available"}
        
        try:
            if global_explanation:
                # Global feature importance
                shap_values = self.shap_explainer.shap_values(instance)
                feature_importance = np.abs(shap_values).mean(0)
            else:
                # Local explanation
                shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))
                feature_importance = np.abs(shap_values).flatten()
            
            # Sort by importance
            importance_indices = np.argsort(feature_importance)[::-1]
            
            return {
                "shap_values": shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                "feature_importance": {
                    self.feature_names[i]: float(feature_importance[i])
                    for i in importance_indices[:10]  # Top 10
                },
                "top_features": [
                    {
                        "feature": self.feature_names[i],
                        "importance": float(feature_importance[i]),
                        "shap_value": float(shap_values.flatten()[i]) if hasattr(shap_values, 'flatten') else float(shap_values[i]),
                    }
                    for i in importance_indices[:10]
                ],
            }
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return {"error": str(e)}
    
    def explain_lime(
        self,
        instance: np.ndarray,
        num_features: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate LIME explanations.
        
        Args:
            instance: Instance to explain
            num_features: Number of top features to explain
            
        Returns:
            Explanation dictionary with LIME explanation
        """
        if not LIME_AVAILABLE or self.lime_explainer is None:
            logger.warning("LIME explanations not available")
            return {"error": "LIME not available"}
        
        try:
            def predict_fn(x):
                """Wrapper for model prediction."""
                if hasattr(self.model, 'predict'):
                    return self.model.predict(x)
                elif hasattr(self.model, '__call__'):
                    return self.model(x)
                else:
                    return np.zeros(len(x))
            
            explanation = self.lime_explainer.explain_instance(
                instance.flatten(),
                predict_fn,
                num_features=num_features,
            )
            
            # Extract explanation data
            explanation_list = explanation.as_list()
            
            return {
                "explanation": explanation_list,
                "top_features": [
                    {
                        "feature": feature,
                        "importance": importance,
                    }
                    for feature, importance in explanation_list
                ],
                "prediction": float(explanation.predicted_value),
            }
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            return {"error": str(e)}
    
    def explain_counterfactual(
        self,
        instance: np.ndarray,
        target_outcome: Optional[float] = None,
        max_changes: int = 5,
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations.
        
        Args:
            instance: Original instance
            target_outcome: Desired outcome (if specified)
            max_changes: Maximum number of features to change
            
        Returns:
            Counterfactual explanation with minimal changes
        """
        # Simplified counterfactual generation
        # In production, use dedicated counterfactual libraries
        
        try:
            current_prediction = self._predict(instance)
            
            # Simple counterfactual: find features that would change prediction
            counterfactuals = []
            
            for i, feature_name in enumerate(self.feature_names[:max_changes]):
                # Create modified instance
                modified = instance.copy()
                modified[i] = modified[i] * 1.1  # 10% increase
                
                new_prediction = self._predict(modified)
                change = new_prediction - current_prediction
                
                counterfactuals.append({
                    "feature": feature_name,
                    "original_value": float(instance[i]),
                    "modified_value": float(modified[i]),
                    "prediction_change": float(change),
                })
            
            return {
                "original_prediction": float(current_prediction),
                "counterfactuals": sorted(counterfactuals, key=lambda x: abs(x["prediction_change"]), reverse=True),
            }
        except Exception as e:
            logger.error(f"Counterfactual explanation failed: {e}")
            return {"error": str(e)}
    
    def _predict(self, instance: np.ndarray) -> float:
        """Helper to get model prediction."""
        try:
            if hasattr(self.model, 'predict'):
                pred = self.model.predict(instance.reshape(1, -1))
                return pred[0] if isinstance(pred, np.ndarray) else pred
            elif hasattr(self.model, '__call__'):
                pred = self.model(instance)
                return float(pred)
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def generate_natural_language_explanation(
        self,
        instance: np.ndarray,
        prediction: Optional[float] = None,
    ) -> str:
        """
        Generate human-readable natural language explanation.
        
        Args:
            instance: Instance to explain
            prediction: Model prediction (if not provided, will be computed)
            
        Returns:
            Natural language explanation string
        """
        if prediction is None:
            prediction = self._predict(instance)
        
        # Get feature importance
        shap_explanation = self.explain_shap(instance)
        
        if "error" in shap_explanation:
            # Fallback to simple explanation
            return f"Model prediction: {prediction:.2f}"
        
        # Build natural language explanation
        top_features = shap_explanation.get("top_features", [])
        
        explanation_parts = [
            f"The traffic control decision resulted in a predicted wait time of {prediction:.2f} seconds.",
            "The key factors influencing this decision were:",
        ]
        
        for i, feature_info in enumerate(top_features[:5], 1):
            feature_name = feature_info["feature"]
            importance = feature_info["importance"]
            shap_val = feature_info.get("shap_value", 0)
            
            direction = "increased" if shap_val > 0 else "decreased"
            explanation_parts.append(
                f"{i}. {feature_name} ({importance:.3f} importance) {direction} the predicted wait time."
            )
        
        return " ".join(explanation_parts)


class ExplainabilityReport:
    """Comprehensive explainability report generator."""
    
    def __init__(self, explainer: EnhancedExplainer):
        """Initialize with explainer."""
        self.explainer = explainer
    
    def generate_comprehensive_report(
        self,
        instance: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report.
        
        Combines SHAP, LIME, and counterfactual explanations.
        """
        report = {
            "instance": instance.tolist(),
            "timestamp": None,  # Will be set
            "explanations": {},
        }
        
        # SHAP explanation
        report["explanations"]["shap"] = self.explainer.explain_shap(instance)
        
        # LIME explanation
        report["explanations"]["lime"] = self.explainer.explain_lime(instance)
        
        # Counterfactual explanation
        report["explanations"]["counterfactual"] = self.explainer.explain_counterfactual(instance)
        
        # Natural language explanation
        report["explanations"]["natural_language"] = self.explainer.generate_natural_language_explanation(instance)
        
        # Feature importance summary
        report["feature_importance_summary"] = self._summarize_importance(
            report["explanations"]
        )
        
        from datetime import datetime
        report["timestamp"] = datetime.now().isoformat()
        
        return report
    
    def _summarize_importance(self, explanations: Dict[str, Any]) -> Dict[str, float]:
        """Summarize feature importance across methods."""
        importance_scores = {}
        
        # Aggregate from SHAP
        if "shap" in explanations and "feature_importance" in explanations["shap"]:
            for feature, importance in explanations["shap"]["feature_importance"].items():
                importance_scores[feature] = importance_scores.get(feature, 0) + importance
        
        # Aggregate from LIME
        if "lime" in explanations and "top_features" in explanations["lime"]:
            for feature_info in explanations["lime"]["top_features"]:
                feature = feature_info["feature"]
                importance = abs(feature_info["importance"])
                importance_scores[feature] = importance_scores.get(feature, 0) + importance
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v / total for k, v in importance_scores.items()}
        
        return dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

