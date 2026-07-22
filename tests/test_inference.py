import pytest
from backend.inference_service import InferenceService

def test_inference_service_initialization():
    # Will fall back to CPU if TensorRT/CUDA unavailable
    service = InferenceService("backend/model.onnx")
    assert service.session is not None
    assert service.input_name is not None
    # We should have active providers
    providers = service.session.get_providers()
    assert len(providers) > 0

def test_inference_service_predict():
    service = InferenceService("backend/model.onnx")
    # Valid feature length is 5
    features = [1.5, 45.2, 1200.0, 10.5, 2.0]
    res = service.predict(features)
    
    assert "error" not in res
    assert "label" in res
    assert "confidence" in res
    assert isinstance(res["label"], int)
    assert 0.0 <= res["confidence"] <= 1.0

def test_inference_service_invalid_features():
    service = InferenceService("backend/model.onnx")
    
    # Missing feature (only 4 passed instead of 5)
    features = [1.5, 45.2, 1200.0, 10.5]
    res = service.predict(features)
    assert "error" in res
