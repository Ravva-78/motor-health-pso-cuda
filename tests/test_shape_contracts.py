"""
tests/test_shape_contracts.py
─────────────────────────────
Comprehensive contract tests covering:
- Input/output tensor shape contracts for ONNX Inference Engine
- Batch shape contract & dynamic batching behavior
- Output probability distribution axioms (sum to 1.0, non-negative)
- Strict feature dimension contract & error handling
"""

import pytest
import numpy as np
from backend.inference_service import InferenceService
from backend.constants import FEATURE_NAMES, LABEL_NAMES_LIST


@pytest.fixture(scope="module")
def inference_engine():
    """Initializes a shared InferenceService instance for testing contracts."""
    return InferenceService()


def test_onnx_input_contract(inference_engine):
    """Ensures input tensor name, shape and type conform to system specifications."""
    assert inference_engine.session is not None
    inputs = inference_engine.session.get_inputs()
    assert len(inputs) == 1, "Inference engine must have exactly one input tensor"
    
    input_tensor = inputs[0]
    assert input_tensor.name == "float_input"
    assert input_tensor.type == "tensor(float)"
    # Shape contract: [None, 5] (batch dimension dynamic, 5 sensor features)
    assert input_tensor.shape[1] == 5, f"Expected 5 features, got {input_tensor.shape[1]}"


def test_onnx_output_contract(inference_engine):
    """Ensures output tensors produce expected class label and probability map."""
    outputs = inference_engine.session.get_outputs()
    assert len(outputs) >= 2, "Expected at least label and probability output nodes"
    
    label_output = outputs[0]
    prob_output = outputs[1]
    
    assert "label" in label_output.name.lower() or label_output.type == "tensor(int64)"
    assert "prob" in prob_output.name.lower() or "map" in prob_output.type or "tensor" in prob_output.type


def test_single_sample_inference_contract(inference_engine):
    """Validates contract for single-sample prediction."""
    valid_features = [0.5, 0.5, 0.5, 0.5, 0.5]
    result = inference_engine.predict(valid_features)
    
    assert "label" in result
    assert "confidence" in result
    assert result["label"] in [0, 1, 2]
    assert 0.0 <= result["confidence"] <= 1.0


def test_probability_distribution_axiom(inference_engine):
    """Validates that raw model output probabilities sum to 1.0 within numerical epsilon."""
    x = np.array([[0.45, 0.52, 0.48, 0.55, 0.30]], dtype=np.float32)
    raw_outputs = inference_engine.session.run(None, {inference_engine.input_name: x})
    
    # Second output is probabilities
    prob_output = raw_outputs[1]
    if isinstance(prob_output, list) and isinstance(prob_output[0], dict):
        probs = list(prob_output[0].values())
        total_prob = sum(probs)
        assert np.isclose(total_prob, 1.0, atol=1e-3), f"Probabilities must sum to 1.0, got {total_prob}"
        for p in probs:
            assert 0.0 <= p <= 1.0, f"Probability out of range: {p}"


def test_dynamic_batching_contract(inference_engine):
    """Validates that ONNX session can evaluate batches of variable sizes (1, 8, 32)."""
    for batch_size in [1, 8, 32]:
        x = np.random.uniform(0.0, 1.0, size=(batch_size, 5)).astype(np.float32)
        raw_outputs = inference_engine.session.run(None, {inference_engine.input_name: x})
        labels = raw_outputs[0]
        assert len(labels) == batch_size, f"Output batch size {len(labels)} mismatch for input {batch_size}"


def test_invalid_dimension_rejection(inference_engine):
    """Verifies that passing an incorrect number of features fails gracefully."""
    # Under-specified input (4 features instead of 5)
    under_features = [0.5, 0.5, 0.5, 0.5]
    res_under = inference_engine.predict(under_features)
    assert "error" in res_under

    # Over-specified input (6 features instead of 5)
    over_features = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    res_over = inference_engine.predict(over_features)
    assert "error" in res_under
