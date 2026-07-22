# Deployment Guide

## Production Considerations
- **MQTT**: Replace the default Eclipse Mosquitto with an enterprise broker if scaling beyond 10,000 req/sec.
- **Docker**: The `docker-compose.yml` provides a blueprint. For production, deploy across Kubernetes clusters.
- **GPU Pass-through**: Ensure your orchestrator (k8s/Docker Swarm) is correctly configured for NVIDIA device plugin.