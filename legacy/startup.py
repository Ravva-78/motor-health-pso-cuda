"""
startup.py
──────────
Auto-trains the model if saved_model.pkl is not found.
This runs when deployed on Streamlit Community Cloud or Render
where the pkl files may not be present.

Called automatically by app.py before loading.
"""

import os
import sys

MODEL_PATH = "backend/saved_model.pkl"

def ensure_model_trained():
    """Train if model doesn't exist — safe to call multiple times."""
    if os.path.exists(MODEL_PATH):
        return True  # Already trained, nothing to do

    print("=" * 55)
    print("  Model not found — running training pipeline...")
    print("  This takes ~3-5 minutes on first deployment.")
    print("=" * 55)

    try:
        # Add project root to path
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from backend.main_training import main
        main()
        print("✅ Training complete — app loading now.")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False


if __name__ == "__main__":
    ensure_model_trained()
