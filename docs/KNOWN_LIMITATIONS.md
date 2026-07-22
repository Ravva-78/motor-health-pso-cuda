# Known Limitations

1. **Docker Validation**: The Docker Compose stack exists but has not been verified under true production loads.
2. **GPU Dependency**: Code gracefully falls back to CPU, but this severely impacts the retraining time of the PSO algorithm and inference latency.
3. **Security**: Currently, MQTT runs without authentication by default (`username: null`). IPC via ZeroMQ is unencrypted TCP.