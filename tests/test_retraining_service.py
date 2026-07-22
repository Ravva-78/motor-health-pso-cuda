import pytest
import time
from unittest.mock import MagicMock, patch
from backend.retraining_service import RetrainingService

def test_retraining_service_success():
    # Mock services
    inf_mock = MagicMock()
    inf_mock.reload_model.return_value = True
    
    shap_mock = MagicMock()
    shap_mock.reload_model.return_value = True
    
    service = RetrainingService(inf_mock, shap_mock)
    
    # Mock subprocess.run to simulate successful training
    with patch('subprocess.run') as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_run.return_value = mock_result
        
        drift_report = {"reason": "Test Drift"}
        service.trigger_retraining(drift_report)
        
        # Wait for background thread to finish
        time.sleep(0.5)
        
        # Verify subprocess was called
        mock_run.assert_called_once()
        
        # Verify hot swap was triggered
        inf_mock.reload_model.assert_called_once()
        shap_mock.reload_model.assert_called_once()

def test_retraining_service_failure():
    # Mock services
    inf_mock = MagicMock()
    shap_mock = MagicMock()
    
    service = RetrainingService(inf_mock, shap_mock)
    
    # Mock subprocess.run to simulate FAILED training
    with patch('subprocess.run') as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 1 # Error
        mock_result.stderr = "Traceback..."
        mock_run.return_value = mock_result
        
        drift_report = {"reason": "Test Failure"}
        service.trigger_retraining(drift_report)
        
        # Wait for background thread to finish
        time.sleep(0.5)
        
        # Verify hot swap was NOT triggered because training failed
        inf_mock.reload_model.assert_not_called()
        shap_mock.reload_model.assert_not_called()
