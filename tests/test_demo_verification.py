"""
tests/test_demo_verification.py
───────────────────────────────
Integration test verifying that the zero-setup quickstart demo runs to completion out of the box.
"""

import subprocess
import sys
import os
import pytest


def test_quickstart_demo_execution():
    """Runs scripts/quickstart_demo.py and verifies that it exits cleanly with code 0."""
    script_path = os.path.join("scripts", "quickstart_demo.py")
    assert os.path.exists(script_path), f"Demo script not found: {script_path}"
    
    result = subprocess.run(
        [sys.executable, script_path, "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30
    )
    
    assert result.returncode == 0, f"Demo failed with stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    assert "AEROFORGE" in result.stdout
    assert "Inference:" in result.stdout
    assert "STEP 1" in result.stdout
    assert "STEP 5" in result.stdout
    assert "AeroForge Edge Pipeline Verification Completed" in result.stdout
