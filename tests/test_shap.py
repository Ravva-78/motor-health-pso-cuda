import pytest
from backend.shap_service import SHAPService

def test_shap_service_initialization():
    service = SHAPService("backend/saved_model.pkl")
    # if it fails due to missing pickle, it will be None.
    # We assume backend/saved_model.pkl exists because main_training.py just ran.
    assert service.explainer is not None

def test_shap_service_explain():
    service = SHAPService("backend/saved_model.pkl")
    features = [1.5, 45.2, 1200.0, 10.5, 2.0]
    # Predict label 0
    res = service.explain(features, predicted_label=0)
    
    assert "error" not in res
    assert "top_features" in res
    assert "feature_contributions" in res
    
    # Check that we got rankings
    assert len(res["top_features"]) <= 3
    assert len(res["feature_contributions"]) == 5
    
    # Keys should match feature names
    for f in service.feature_names:
        assert f in res["feature_contributions"]
