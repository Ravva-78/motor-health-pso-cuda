import os
import yaml
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from backend.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    """
    Centralized configuration management.
    Loads config.yaml, applies sensible defaults, and allows environment variable overrides.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> dict:
        cfg = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    cfg = yaml.safe_load(f) or {}
            except Exception as e:
                logger.error(f"Error loading {self.config_path}: {e}")
        else:
            logger.warning(f"Configuration file {self.config_path} not found. Using defaults.")
        return cfg

    def get_pso_config(self) -> dict:
        pso_cfg = self.config.get('pso', {})
        return {
            'n_particles': int(os.environ.get('PSO_PARTICLES', pso_cfg.get('n_particles', 20))),
            'max_iter': int(os.environ.get('PSO_MAX_ITER', pso_cfg.get('max_iter', 20))),
            'inertia_start': float(os.environ.get('PSO_INERTIA_START', pso_cfg.get('inertia_start', 0.9))),
            'cognitive': float(os.environ.get('PSO_COGNITIVE', pso_cfg.get('cognitive', 1.5))),
            'social': float(os.environ.get('PSO_SOCIAL', pso_cfg.get('social', 1.5))),
        }

    def get_model_config(self) -> dict:
        mod_cfg = self.config.get('model', {})
        return {
            'random_state': int(os.environ.get('MODEL_RANDOM_STATE', mod_cfg.get('random_state', 42))),
            'n_jobs': int(os.environ.get('MODEL_N_JOBS', mod_cfg.get('n_jobs', -1))),
            'cv_folds': int(os.environ.get('MODEL_CV_FOLDS', mod_cfg.get('cv_folds', 5))),
        }
        
    def get_cuda_config(self) -> dict:
        cuda_cfg = self.config.get('cuda', {})
        return {
            'threads_per_block': int(os.environ.get('CUDA_THREADS_PER_BLOCK', cuda_cfg.get('threads_per_block', 256))),
            'benchmark_repeats': int(os.environ.get('CUDA_BENCHMARK_REPEATS', cuda_cfg.get('benchmark_repeats', 50))),
        }

    def get_mqtt_config(self) -> dict:
        mqtt_cfg = self.config.get('mqtt', {})
        return {
            'broker': str(os.environ.get('MQTT_BROKER', mqtt_cfg.get('broker', 'localhost'))),
            'port': int(os.environ.get('MQTT_PORT', mqtt_cfg.get('port', 1883))),
            'username': os.environ.get('MQTT_USERNAME', mqtt_cfg.get('username', None)),
            'password': os.environ.get('MQTT_PASSWORD', mqtt_cfg.get('password', None)),
            'qos': int(os.environ.get('MQTT_QOS', mqtt_cfg.get('qos', 1))),
            'topics': mqtt_cfg.get('topics', ['aeroforge/telemetry']),
            'client_id': os.environ.get('MQTT_CLIENT_ID', mqtt_cfg.get('client_id', 'aeroforge_backend')),
            'min_reconnect_delay': int(os.environ.get('MQTT_MIN_RECONNECT_DELAY', mqtt_cfg.get('min_reconnect_delay', 1))),
            'max_reconnect_delay': int(os.environ.get('MQTT_MAX_RECONNECT_DELAY', mqtt_cfg.get('max_reconnect_delay', 120))),
        }

    def get_ring_buffer_config(self) -> dict:
        rb_cfg = self.config.get('ring_buffer', {})
        return {
            'capacity': int(os.environ.get('RING_BUFFER_CAPACITY', rb_cfg.get('capacity', 4096)))
        }

    def get_ipc_config(self) -> dict:
        ipc_cfg = self.config.get('ipc', {})
        return {
            'address': str(os.environ.get('IPC_ADDRESS', ipc_cfg.get('address', 'tcp://127.0.0.1:5555'))),
            'timeout_ms': int(os.environ.get('IPC_TIMEOUT_MS', ipc_cfg.get('timeout_ms', 2000)))
        }

    def get_paths(self) -> dict:
        paths_cfg = self.config.get('paths', {})
        return {
            'data': os.environ.get('PATH_DATA', paths_cfg.get('data', 'data/predictive_maintenance.csv')),
            'model': os.environ.get('PATH_MODEL', paths_cfg.get('model', 'backend/saved_model.pkl')),
            'scaler': os.environ.get('PATH_SCALER', paths_cfg.get('scaler', 'backend/scaler.pkl')),
            'results': os.environ.get('PATH_RESULTS', paths_cfg.get('results', 'backend/training_results.pkl')),
            'metrics_csv': os.environ.get('PATH_METRICS_CSV', paths_cfg.get('metrics_csv', 'results/metrics_export.csv')),
        }

config = ConfigManager()
