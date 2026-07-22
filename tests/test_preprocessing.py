import pytest
from backend.preprocessing import load_and_preprocess

def test_load_and_preprocess_success(dummy_csv_path):
    # We pass use_smote=False to avoid k_neighbors errors on tiny dummy datasets
    X_train, X_test, y_train, y_test, feature_names, scaler, df_clean = load_and_preprocess(dummy_csv_path, use_smote=False)
    
    assert X_train is not None
    assert X_test is not None
    assert len(feature_names) == 5
    
    # 10 samples total, 80/20 split means 8 train, 2 test
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
    
    # Ensure health mapping worked
    assert 'health' in df_clean.columns
    assert 'label' in df_clean.columns

def test_load_and_preprocess_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_preprocess("completely_nonexistent_file.csv")
