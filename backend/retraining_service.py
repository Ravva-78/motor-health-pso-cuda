import threading
import subprocess
import sys
import os
from backend.logger import get_logger

logger = get_logger(__name__)

class RetrainingService:
    """
    Listens for drift events and safely orchestrates the async execution of the PSO pipeline.
    Executes the heavy ML workload in a detached subprocess to preserve Daemon memory and latency.
    """
    def __init__(self, inference_service, shap_service):
        self.inference_service = inference_service
        self.shap_service = shap_service
        self._retraining_lock = threading.Lock()
        
    def trigger_retraining(self, drift_report: dict):
        if self._retraining_lock.locked():
            logger.info("Retraining is already in progress. Ignoring drift trigger.")
            return
            
        # Spawn daemon thread so it doesn't block IPC or telemetry loops
        t = threading.Thread(target=self._run_retraining, args=(drift_report,), daemon=True)
        t.start()
        
    def _run_retraining(self, drift_report: dict):
        with self._retraining_lock:
            logger.info(f"Starting async PSO retraining due to drift: {drift_report.get('reason', 'Unknown')}")
            try:
                # Set up environment to ensure main_training.py finds the 'backend' package
                env = os.environ.copy()
                project_root = os.path.dirname(os.path.dirname(__file__))
                env['PYTHONPATH'] = project_root
                
                # Execute pipeline synchronously within this background thread
                result = subprocess.run(
                    [sys.executable, "backend/main_training.py"],
                    capture_output=True,
                    text=True,
                    env=env
                )
                
                if result.returncode == 0:
                    logger.info("Async PSO retraining completed successfully. Triggering atomic model swap.")
                    
                    # Atomic hot swaps
                    inf_success = self.inference_service.reload_model()
                    shap_success = self.shap_service.reload_model()
                    
                    if inf_success and shap_success:
                        logger.info("✅ [SUCCESS] Production models hot-swapped seamlessly.")
                    else:
                        logger.error("Hot-swap failed for one or more services. Check logs.")
                else:
                    logger.error(f"Async retraining failed with exit code {result.returncode}.")
                    # Dump the last 20 lines of the error to the log
                    err_lines = result.stderr.split('\n')[-20:]
                    logger.error(f"Error tail:\n{chr(10).join(err_lines)}")
                    
            except Exception as e:
                logger.error(f"Exception during async retraining execution: {e}")
